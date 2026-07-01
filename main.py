import asyncio
import json
import os
import random
import re
import time
from collections import deque

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger, AstrBotConfig


@register(
    "autoreply_judge",
    "StarBot",
    "LLM智能判断群聊消息是否需要自动回复",
    "1.2.2",
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
        self._judging_groups: set[str] = set()
        self._judging_groups_lock = asyncio.Lock()
        self._judged_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        self._switch_lock = asyncio.Lock()
        # 记录原始 is_at_or_wake_command 状态（在 on_group_message 修改之前）
        # key = "group_id:msg" , value = 原始 is_at_or_wake_command
        self._original_at_state: dict[str, bool] = {}

    async def initialize(self):
        self._load_switches()
        logger.info(f"判断插件已加载 v1.2.2，已恢复 {len(self._group_switch)} 个群开关状态")

    def _load_switches(self):
        try:
            with open(self._switch_file, "r", encoding="utf-8") as f:
                self._group_switch = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._group_switch = {}

    def _save_switches(self):
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
        async with self._switch_lock:
            if new_state is not None:
                self._group_switch[group_id] = new_state
            else:
                self._group_switch[group_id] = not self._group_switch.get(group_id, True)
            self._save_switches()
            status = "已开启" if self._group_switch[group_id] else "已关闭"
        yield event.plain_result(f"本群自动回复判断：{status}")

    # ========== 群消息历史记录（v1.1 架构回归）==========

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """记录群聊历史，并确保非@消息也能触发LLM判断"""
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

        # ★ 修复：在修改 is_at_or_wake_command 之前，保存原始的@状态
        key = f"{group_id}:{msg}"
        # 仅当没有记录过时才写入，避免重复消息覆盖
        if key not in self._original_at_state:
            self._original_at_state[key] = event.is_at_or_wake_command

        # 关键：让非@群消息也能触发 LLM Agent → on_llm_request
        if not event.is_at_or_wake_command:
            event.is_at_or_wake_command = True

    # ========== LLM 请求拦截判断（核心）==========

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """判断是否要回复；拦截靠 stop_event"""
        try:
            if not self.config.get("enabled", True):
                return
            group_id = self._get_group_id(event)
            if not group_id:
                return
            msg = (event.message_str or "").strip()
            if not msg or msg.startswith("/"):
                return
            if not self._group_switch.get(group_id, True):
                return

            cache_key = f"{group_id}:{msg}:{int(time.time()/60)}"

            async with self._judged_lock:
                if cache_key in self._judged:
                    entry = self._judged[cache_key]
                    if entry["block"]:
                        event.stop_event()
                        logger.info(f"缓存阻断 | {group_id} | {msg[:40]}")
                    return

            # ★ 修复：读取保存的原始@状态判断是否为真正的@消息
            #   同时兼容旧版格式 [At: 作为兜底
            key = f"{group_id}:{msg}"
            is_original_at = self._original_at_state.pop(key, None)
            if is_original_at is None:
                # 没有记录（极少边界情况），走 [At: 格式兜底
                is_original_at = "[At:" in msg

            if is_original_at:
                # 真正的@消息 → 跳过LLM判断，直接放行
                async with self._judged_lock:
                    self._judged[cache_key] = {"block": False, "time": time.time()}
                logger.info(f"@放行 | {group_id} | {msg[:40]}")
                return

            async with self._judging_groups_lock:
                if group_id in self._judging_groups:
                    return
                self._judging_groups.add(group_id)

            try:
                sender = event.get_sender_name() or "未知"
                result = await self._llm_judge(event, group_id, msg, sender)
                if result is None:
                    return

                should_reply = result.get("should_reply", True)
                confidence = result.get("confidence", 0)
                reason = result.get("reason", "")

                async with self._judged_lock:
                    if cache_key in self._judged:
                        entry = self._judged[cache_key]
                        if entry["block"]:
                            event.stop_event()
                            logger.info(f"缓存阻断(double) | {group_id} | {msg[:40]}")
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
            finally:
                async with self._judging_groups_lock:
                    self._judging_groups.discard(group_id)
        except Exception as e:
            logger.error(f"🔴 on_llm_request 异常: {e}", exc_info=True)

    # ========== 工具方法 ==========

    def _cleanup_expired_cache(self):
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

            context_str = ""
            async with self._history_lock:
                if group_id in self._history:
                    ctx_size = max(0, self.config.get("context_size", 3))
                    recent = list(self._history[group_id])[-ctx_size:] if ctx_size > 0 else []
                    lines = [f"{h[0]}: {h[1]}" for h in recent if h[1] != msg]
                    if lines:
                        context_str = "\n".join(lines)

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
            result = self._parse_response(text)
            if result is None:
                logger.warning(f"⚠️ 判断结果解析失败 | {group_id} | {msg[:40]} | 原始响应: {text[:200]}")
            return result
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
        placeholders = {}

        def _replace_strings(m):
            key = f"\x00STR_{len(placeholders)}\x00"
            placeholders[key] = m.group(0)
            return key

        safe = re.sub(r'"(?:[^"\\\\]|\\\\.)*"', _replace_strings, text)
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
        result = self._extract_json(text)
        if result:
            return result
        # 文本兜底：匹配 should_reply=true/false 或单纯 true/false
        t = text.strip().lower()
        m_sr = re.search(r"should_reply\s*[=:]\s*(true|false)", t)
        if m_sr:
            return {
                "should_reply": m_sr.group(1) == "true",
                "confidence": 50,
                "reason": "",
            }
        if t in ("true", "false"):
            return {
                "should_reply": t == "true",
                "confidence": 50,
                "reason": "",
            }
        return None

    def _get_group_id(self, event):
        """从 unified_msg_origin 提取群ID（v1.1 方案 + 多源回退）"""
        umo = event.unified_msg_origin or ""
        parts = umo.split(":")
        if len(parts) >= 3 and "Group" in parts[1]:
            return parts[-1]
        # 回退：raw_event / message_obj
        try:
            raw = getattr(event, "raw_event", None) or {}
            for key in ("group_id", "group_uin"):
                val = raw.get(key)
                if val is not None and val != "":
                    return str(val)
        except Exception:
            pass
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