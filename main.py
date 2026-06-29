import asyncio
import json
import logging
import os
import random
import re
import time
from builtins import GeneratorExit as _GeneratorExit
from collections import deque

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger, AstrBotConfig


@register(
    "autoreply_judge",
    "StarBot",
    "LLM智能判断群聊消息是否需要自动回复",
    "1.1.1",
)
class AutoReplyJudgePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._group_switch = {}
        self._switch_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_group_switches.json"
        )
        self._history = {}
        self._judged = {}
        self._cache_ttl = 120
        self._cleanup_counter = 0
        self._filter_added = False
        self._judging_groups: set[str] = set()
        self._judging_groups_lock = asyncio.Lock()  # 保护 _judging_groups 的原子性 check-and-set
        # 并发锁：保护 _judged 字典（多 group 并发写入 + 遍历清理）
        self._judged_lock = asyncio.Lock()
        # 并发锁：保护 _history 字典（on_group_message 写入 vs _llm_judge 读取）
        self._history_lock = asyncio.Lock()
        # 并发锁：保护 _group_switch 字典 + 开关文件写入（/reply 指令并发安全）
        self._switch_lock = asyncio.Lock()

    async def initialize(self):
        self._load_switches()
        if not self._filter_added:
            for name in ("astrbot.main", "astrbot"):
                logging.getLogger(name).addFilter(
                    lambda r: not (
                        "GeneratorExit" in (r.getMessage() + (r.exc_text or ""))
                        or (r.exc_info and r.exc_info[1] and isinstance(r.exc_info[1], _GeneratorExit))
                        or "主动回复失败" in r.getMessage()
                    )
                )
            self._filter_added = True
        logger.info(f"判断插件已加载 v1.1.1，已恢复 {len(self._group_switch)} 个群开关状态")

    def _load_switches(self):
        try:
            with open(self._switch_file, "r", encoding="utf-8") as f:
                self._group_switch = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._group_switch = {}

    def _save_switches(self):
        """原子写入群开关状态：写临时文件 → os.replace() 原子重命名，防止崩溃导致文件损坏"""
        try:
            tmp = self._switch_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._group_switch, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._switch_file)
        except Exception as e:
            logger.error(f"保存群开关状态失败: {e}")

    @filter.command("reply")
    async def toggle_reply(self, event: AstrMessageEvent):
        group_id = self._get_group_id(event)
        if not group_id:
            yield event.plain_result("请在群聊中使用此指令")
            return
        args = (event.message_str or "").strip().split()
        # 先验证参数合法性（无需持锁，避免 yield 在锁内）
        new_state = None
        if len(args) >= 2:
            arg = args[1].lower()
            if arg in ("true", "1", "on", "开", "开启"):
                new_state = True
            elif arg in ("false", "0", "off", "关", "关闭"):
                new_state = False
            else:
                yield event.plain_result("参数错误，可用：on/off/true/false/开/关")
                return
        # 加锁执行状态变更和文件写入（锁内不 yield）
        async with self._switch_lock:
            if new_state is not None:
                self._group_switch[group_id] = new_state
            else:
                self._group_switch[group_id] = not self._group_switch.get(group_id, True)
            self._save_switches()
            status = "已开启" if self._group_switch[group_id] else "已关闭"
        # yield 在锁外，安全
        yield event.plain_result(f"本群自动回复判断：{status}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        if not self.config.get("enabled", True):
            return
        msg = (event.message_str or "").strip()
        if msg.startswith("/"):
            return
        group_id = self._get_group_id(event)
        if not group_id:
            return
        if not self._group_switch.get(group_id, True):
            return
        sender = event.get_sender_name() or "未知"
        await self._record_history(group_id, sender, msg)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.config.get("enabled", True):
            return
        group_id = self._get_group_id(event)
        if not group_id:
            return
        # 原子性 check-and-set，防止同一群两条消息同时进入 LLM 判断
        async with self._judging_groups_lock:
            if group_id in self._judging_groups:
                logger.debug(f"同群判断进行中，跳过 | {group_id} | {(event.message_str or '')[:40]}")
                return
            self._judging_groups.add(group_id)
        try:
            msg = (event.message_str or "").strip()
            if not msg or msg.startswith("/"):
                return
            if not self._group_switch.get(group_id, True):
                return

            cache_key = f"{group_id}:{msg}:{int(time.time()/60)}"

            # ---- 加锁：_judged 缓存读取 ----
            async with self._judged_lock:
                if cache_key in self._judged:
                    entry = self._judged[cache_key]
                    if entry["block"]:
                        event.stop_event()
                        logger.info(f"缓存阻断 | {group_id} | {msg[:40]}")
                    return
            # ---- 锁释放 ----

            sender = event.get_sender_name() or "未知"
            result = await self._llm_judge(event, group_id, msg, sender)
            if result is None:
                return

            should_reply = result.get("should_reply", True)
            confidence = result.get("confidence", 0)
            reason = result.get("reason", "")

            # ---- 加锁：_judged 缓存写入 + 定期清理 ----
            async with self._judged_lock:
                # double-check：防止在 LLM 调用期间其他协程已写入相同缓存
                if cache_key in self._judged:
                    entry = self._judged[cache_key]
                    if entry["block"]:
                        event.stop_event()
                        logger.info(f"缓存阻断(double-check) | {group_id} | {msg[:40]}")
                    return

                self._cleanup_counter = (self._cleanup_counter + 1) % 50
                if self._cleanup_counter == 0:
                    self._cleanup_expired_cache()

                if not should_reply:
                    chance = max(0, min(100, self.config.get("reply_chance", 20)))
                    if random.randint(1, 100) > chance:
                        self._judged[cache_key] = {"block": True, "time": time.time()}
                        event.stop_event()
                        logger.info(f"拦截 | {group_id} | 置信度:{confidence} | {reason} | {msg[:40]}")
                        return
                    self._judged[cache_key] = {"block": False, "time": time.time()}
                    logger.info(f"概率放行 | {group_id} | 置信度:{confidence} | 原因:{reason} | {msg[:40]}")
                    return

                self._judged[cache_key] = {"block": False, "time": time.time()}
                logger.info(f"LLM放行 | {group_id} | 置信度:{confidence} | {reason} | {msg[:40]}")
            # ---- 锁释放 ----
        finally:
            async with self._judging_groups_lock:
                self._judging_groups.discard(group_id)

    def _cleanup_expired_cache(self):
        """清理过期缓存。⚠️ 调用方必须已持有 self._judged_lock"""
        try:
            now = time.time()
            expired = [k for k, v in self._judged.items() if now - v.get("time", 0) > self._cache_ttl]
            for k in expired:
                del self._judged[k]
            if expired:
                logger.debug(f"缓存清理: 移除 {len(expired)} 条过期记录")
        except Exception as e:
            logger.error(f"缓存清理异常: {e}")

    async def _llm_judge(self, event, group_id, msg, sender):
        try:
            prompt = self.config.get("judge_prompt", "")
            if not prompt:
                return None

            # ---- 加锁：_history 读取 ----
            context_str = ""
            async with self._history_lock:
                if group_id in self._history:
                    ctx_size = max(0, self.config.get("context_size", 3))
                    recent = list(self._history[group_id])[-ctx_size:] if ctx_size > 0 else []
                    lines = [f"{h[0]}: {h[1]}" for h in recent if h[1] != msg]
                    if lines:
                        context_str = "\n".join(lines)
            # ---- 锁释放 ----

            prompt = re.sub(
                r"\{message\}|\{context\}|\{sender\}",
                lambda m: {
                    "{message}": msg,
                    "{context}": context_str or "（无）",
                    "{sender}": sender,
                }[m.group(0)],
                prompt,
            )
            prov = await self._get_judge_provider(event)
            if not prov:
                return None
            resp = await asyncio.wait_for(
                prov.text_chat(prompt=prompt, context=[]),
                timeout=15.0,
            )
            if not resp:
                return None
            if hasattr(resp, "completion_text"):
                text = resp.completion_text
            else:
                text = str(resp)
            return self._parse_response(text)
        except asyncio.TimeoutError:
            logger.warning(f"LLM判断超时(15s)，放行 | {group_id} | {msg[:40]}")
            return None
        except Exception as e:
            logger.error(f"LLM判断异常: {e}")
            return None

    async def _get_judge_provider(self, event):
        provider_id = self.config.get("judge_provider", "").strip()
        if provider_id:
            prov = self.context.get_provider_by_id(provider_id=provider_id)
            if prov:
                return prov
            logger.warning(f"未找到提供商 {provider_id}，回退对话模型")
        return self.context.get_using_provider(umo=event.unified_msg_origin)

    @staticmethod
    def _normalize_result(data):
        if not isinstance(data, dict):
            return None
        should_reply = data.get("should_reply", True)
        # LLM 有时输出字符串 "false"/"true" 而非 JSON 布尔值
        if isinstance(should_reply, str):
            should_reply = should_reply.lower() in ("true", "1", "yes")
        elif isinstance(should_reply, (int, float)):
            should_reply = should_reply > 0
        return {
            "should_reply": bool(should_reply),
            "confidence": int(data.get("confidence", 50)),
            "reason": str(data.get("reason", "")),
        }

    @classmethod
    def _extract_json(cls, text):
        """用栈匹配法从文本中提取第一个完整的最外层JSON对象（字符串感知）"""
        brace_stack = []
        json_start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if not brace_stack:
                    json_start = i
                brace_stack.append(i)
            elif ch == "}":
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and json_start >= 0:
                        candidate = text[json_start : i + 1]
                        try:
                            data = json.loads(candidate)
                            return cls._normalize_result(data)
                        except json.JSONDecodeError:
                            fixed = cls._fix_trailing_commas(candidate)
                            try:
                                data = json.loads(fixed)
                                return cls._normalize_result(data)
                            except json.JSONDecodeError:
                                json_start = -1
                                continue
        return None

    @staticmethod
    def _fix_trailing_commas(text):
        """智能修复JSON尾部逗号：用占位符替换字符串后再修复，避免误伤字符串内内容"""
        placeholders = {}

        def _replace_strings(m):
            key = f"\x00STR_{len(placeholders)}\x00"
            placeholders[key] = m.group(0)
            return key

        safe = re.sub(r'"(?:[^"\\]|\\.)*"', _replace_strings, text)
        safe = re.sub(r",\s*([}\]])", r"\1", safe)
        for key, val in placeholders.items():
            safe = safe.replace(key, val)
        return safe

    def _parse_response(self, text):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if m:
            content = m.group(1).strip()
            try:
                data = json.loads(content)
                return self._normalize_result(data)
            except json.JSONDecodeError:
                pass
            result = self._extract_json(content)
            if result:
                return result
        return self._extract_json(text)

    def _get_group_id(self, event):
        """从事件中提取群ID，非群消息返回 None"""
        # AstrMessageEvent 标准 API：群消息返回群ID，非群消息返回空字符串
        gid = event.get_group_id()
        if gid is not None and gid != "":
            return gid
        return None

    async def _record_history(self, group_id, sender, msg):
        async with self._history_lock:
            if group_id not in self._history:
                maxlen = max(1, self.config.get("history_maxlen", 10))
                self._history[group_id] = deque(maxlen=maxlen)
            self._history[group_id].append((sender, msg))

    async def terminate(self):
        async with self._switch_lock:
            self._save_switches()
        logger.info("插件已卸载，群开关已保存")
