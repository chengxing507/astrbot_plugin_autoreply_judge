import json
import logging
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
    "1.0.0",
)
class AutoReplyJudgePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._group_switch = {}
        self._history = {}
        self._judged = {}

    async def initialize(self):
        for name in ("astrbot.main", "astrbot"):
            logging.getLogger(name).addFilter(
                lambda r: not (
                    "GeneratorExit" in (r.getMessage() + (r.exc_text or ""))
                    or (r.exc_info and r.exc_info[1] and isinstance(r.exc_info[1], _GeneratorExit))
                    or "主动回复失败" in r.getMessage()
                )
            )
        logger.info("判断插件已加载")

    @filter.command("reply")
    async def toggle_reply(self, event: AstrMessageEvent):
        group_id = self._get_group_id(event)
        if not group_id:
            yield event.plain_result("请在群聊中使用此指令")
            return
        args = (event.message_str or "").strip().split()
        current = self._group_switch.get(group_id, True)
        if len(args) >= 2:
            arg = args[1].lower()
            if arg in ("true", "1", "on", "开", "开启"):
                self._group_switch[group_id] = True
            elif arg in ("false", "0", "off", "关", "关闭"):
                self._group_switch[group_id] = False
            else:
                yield event.plain_result("参数错误")
                return
        else:
            self._group_switch[group_id] = not current
        status = "已开启" if self._group_switch[group_id] else "已关闭"
        yield event.plain_result("本群自动回复判断：" + status)

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
        self._record_history(group_id, sender, msg)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        group_id = self._get_group_id(event)
        if not group_id:
            return
        msg = (event.message_str or "").strip()
        if not msg or msg.startswith("/"):
            return
        if not self._group_switch.get(group_id, True):
            return

        cache_key = f"{group_id}:{msg}:{int(time.time()/60)}"
        if cache_key in self._judged:
            if self._judged[cache_key]:
                event.stop_event()
                logger.info("缓存阻断 | " + str(group_id) + " | " + msg[:40])
            return

        sender = event.get_sender_name() or "未知"
        result = await self._llm_judge(event, group_id, msg, sender)
        if result is None:
            return

        should_reply = result.get("should_reply", True)
        confidence = result.get("confidence", 0)
        reason = result.get("reason", "")

        if not should_reply:
            chance = max(0, min(100, self.config.get("reply_chance", 20)))
            if random.randint(1, 100) > chance:
                self._judged[cache_key] = True
                event.stop_event()
                logger.info("拦截 | " + str(group_id) + " | 置信度:" + str(confidence) + " | " + reason + " | " + msg[:40])
                return
            self._judged[cache_key] = False
            logger.info("概率放行 | " + str(group_id) + " | 原因:" + reason + " | " + msg[:40])
            return

        self._judged[cache_key] = False
        logger.info("LLM放行 | " + str(group_id) + " | 置信度:" + str(confidence) + " | " + reason + " | " + msg[:40])

    async def _llm_judge(self, event, group_id, msg, sender):
        try:
            prompt = self.config.get("judge_prompt", "")
            if not prompt:
                return None
            context_str = ""
            if group_id in self._history:
                ctx_size = max(0, self.config.get("context_size", 3))
                recent = list(self._history[group_id])[-ctx_size:] if ctx_size > 0 else []
                lines = [f"{h[0]}: {h[1]}" for h in recent if h[1] != msg]
                if lines:
                    context_str = "\n".join(lines)
            prompt = prompt.replace("{message}", msg)
            prompt = prompt.replace("{context}", context_str or "（无）")
            prompt = prompt.replace("{sender}", sender)
            prov = await self._get_judge_provider(event)
            if not prov:
                return None
            resp = await prov.text_chat(prompt=prompt, context=[])
            if not resp:
                return None
            text = ""
            if hasattr(resp, "completion_text"):
                text = resp.completion_text
            else:
                text = str(resp)
            return self._parse_response(text)
        except Exception as e:
            logger.error("LLM判断异常: " + str(e))
            return None

    async def _get_judge_provider(self, event):
        provider_id = self.config.get("judge_provider", "").strip()
        if provider_id:
            prov = self.context.get_provider_by_id(provider_id=provider_id)
            if prov:
                return prov
            logger.warning("未找到提供商 " + provider_id + "，回退对话模型")
        return self.context.get_using_provider(umo=event.unified_msg_origin)

    def _parse_response(self, text):
        m = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            m = re.search(r'\{[^{}]*\}', text)
            if m:
                text = m.group(0)
        try:
            data = json.loads(text)
            return {
                "should_reply": bool(data.get("should_reply", True)),
                "confidence": int(data.get("confidence", 50)),
                "reason": str(data.get("reason", "")),
            }
        except Exception:
            return None

    def _get_group_id(self, event):
        umo = event.unified_msg_origin or ""
        parts = umo.split(":")
        if len(parts) >= 3:
            return parts[-1]
        return None

    def _record_history(self, group_id, sender, msg):
        if group_id not in self._history:
            self._history[group_id] = deque(maxlen=10)
        self._history[group_id].append((sender, msg))

    async def terminate(self):
        logger.info("插件已卸载")