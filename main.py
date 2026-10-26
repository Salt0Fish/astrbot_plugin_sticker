"""表情包 + 语音 + 消息合并（私聊友好）。

三件事：
1. 把模型的多行回复合并成 1~2 条消息（确定性规则）
2. 模型写 [[表情:标签]] 时附一张本地表情包；分类没有就从总库随机
3. 每 N 条回复用本地 GPT-SoVITS 合成一条语音（可降调、放慢、加首尾留白）

命令：
  /表情 开心        直接发一张
  /存表情 开心      引用图片后发这条，存进该分类
  /表情列表         查看库存
  /语音 文本        手动合成一条语音
"""

import hashlib
import random
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain, Record

PLUGIN_DIR = Path(__file__).parent
DATA_DIR = PLUGIN_DIR / "data"

# ---- 运行期配置（在 __init__ 里从 AstrBot 配置读取）----
CFG: dict = {}

MARKER_RE = re.compile(r"\[\[\s*(?:表情|sticker)\s*[:：]\s*([^\[\]]{1,12})\s*\]\]")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
LABELS = ["开心", "害羞", "生气", "委屈", "无语", "亲亲", "困"]

AI_SPEAK = [
    r"我理解你的感受", r"我能理解你的", r"作为一个?AI", r"希望能帮到你", r"希望对你有所帮助",
    r"如果你愿意", r"首先[，,]", r"其次[，,]", r"总之", r"总而言之", r"总的来说",
    r"别担心[，,]?一切都会", r"你的感受(很|非常)?重要", r"我很乐意", r"让我们",
    r"建议你", r"需要注意的是", r"听起来你", r"你愿意(和|跟)我(说|聊)",
    r"需要我帮你", r"有什么(可以|能)帮", r"我可以帮你", r"愿你", r"加油哦",
    r"我在这里(陪|支持)", r"你不用(担心|害怕)", r"慢慢来", r"照顾好自己",
]


# ---------------- 路径与配置 ----------------
def _p(value: str, default: Path) -> Path:
    """空字符串时用默认值。"""
    return Path(value) if value else default


def _sticker_root() -> Path:
    return _p(CFG.get("sticker_root", ""), DATA_DIR / "stickers")


def _fallback_dir() -> Path:
    return _sticker_root() / (CFG.get("sticker_fallback_dir") or "表情包库")


def _voice_dir() -> Path:
    return _p(CFG.get("voice_dir", ""), DATA_DIR / "voice")


def _tuned_dir() -> Path:
    d = DATA_DIR / "tuned"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ffmpeg() -> str:
    return CFG.get("ffmpeg_path") or "ffmpeg"


# ---------------- 表情包 ----------------
def _images(tag: str) -> list[Path]:
    d = _sticker_root() / tag
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT]


def _pick(tag: str) -> Path | None:
    files = _images(tag)
    if files:
        return random.choice(files)
    fb = _fallback_dir()
    if fb.is_dir():
        pool = [p for p in fb.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        if pool:
            return random.choice(pool)
    return None


# ---------------- 文本处理 ----------------
def _strip_ai(text: str) -> str:
    if not CFG.get("enable_ai_speak_filter", True):
        return text
    sentences = re.split(r"(?<=[。！？!?\n])", text)
    kept = []
    for s in sentences:
        if not s.strip():
            continue
        if any(re.search(p, s) for p in AI_SPEAK):
            continue
        kept.append(s.strip())
    return "".join(kept)


def _split_text(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return lines or [""]
    limit = int(CFG.get("one_message_limit", 26))
    max_parts = max(1, int(CFG.get("max_parts", 2)))
    if len("".join(lines)) <= limit:
        return ["，".join(lines)]

    total = sum(len(ln) for ln in lines)
    half = total / max_parts
    parts: list[str] = []
    buf = ""
    acc = 0
    for ln in lines:
        if len(parts) < max_parts - 1 and acc + len(ln) > half and buf:
            parts.append(buf)
            buf = ln
            acc = len(ln)
        else:
            buf = buf + ln if not buf else buf + "，" + ln
            acc += len(ln)
    if buf:
        parts.append(buf)
    return parts


def _clean_parts(parts: list[str]) -> list[str]:
    out: list[str] = []
    for p in parts:
        p = _strip_ai(p)
        p = p.replace("。", "，").strip("， ")
        if len(p) > 14 and "，" in p:
            head, tail = p.split("，", 1)
            out.extend([head, tail.strip("， ")])
        elif p:
            out.append(p)
    return [x for x in out if x.strip()][:3]


# ---------------- 语音 ----------------
def _synth(text: str) -> Path | None:
    url = CFG.get("tts_url") or ""
    ref = CFG.get("voice_ref") or ""
    if not url or not ref:
        return None
    body = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": ref,
        "prompt_text": CFG.get("voice_prompt") or "",
        "prompt_lang": "zh",
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
    }
    req = urllib.request.Request(
        url, data=__import__("json").dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        if len(data) < 2000:
            return None
        out_dir = _voice_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{int(time.time() * 1000)}.wav"
        out.write_bytes(data)
        return _tune(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[voice] 合成失败: {exc}")
        return None


def _tune(src: Path) -> Path:
    """降调 + 放慢 + 首尾留白。"""
    pitch = float(CFG.get("voice_pitch_semitone", -1.0))
    speed = float(CFG.get("voice_speed", 0.92))
    head = int(CFG.get("voice_pad_head_ms", 260))
    tail = float(CFG.get("voice_pad_tail_s", 0.4))

    factor = 2 ** (pitch / 12)
    tempo = speed / factor
    dst = _tuned_dir() / (src.stem + "_t.wav")
    chain = (
        f"asetrate=32000*{factor:.6f},aresample=32000,atempo={tempo:.6f},"
        f"adelay={head}|{head},apad=pad_dur={tail}"
    )
    try:
        r = subprocess.run(
            [_ffmpeg(), "-y", "-i", str(src), "-filter:a", chain, str(dst)],
            capture_output=True, timeout=30,
        )
        if dst.exists() and dst.stat().st_size > 2000:
            src.unlink(missing_ok=True)
            return dst
        logger.warning(f"[voice] 调音未产出 rc={r.returncode}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[voice] 调音失败: {exc}")
    return src


# ---------------- 插件主体 ----------------
class Main(star.Star):
    def __init__(self, context: star.Context, config: AstrBotConfig) -> None:
        self.context = context
        CFG.clear()
        CFG.update(dict(config))
        self._since: dict[str, int] = {}
        self._voice: dict[str, int] = {}
        _tuned_dir()

    @filter.on_decorating_result()
    async def decorate_result(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if not result or not result.chain:
            return
        try:
            if not result.is_model_result():
                return
        except Exception:
            pass

        plains = [c for c in result.chain if isinstance(c, Plain)]
        others = [c for c in result.chain if not isinstance(c, Plain)]
        text = "".join((c.text or "") for c in plains)

        tags = MARKER_RE.findall(text)
        text = MARKER_RE.sub("", text)
        parts = _clean_parts([p for p in _split_text(text) if p])

        images: list[Image] = []
        umo = getattr(event, "unified_msg_origin", "") or ""
        cooldown = int(CFG.get("sticker_cooldown", 4))
        since = self._since.get(umo, cooldown)
        if tags:
            if since >= cooldown:
                path = _pick(tags[0].strip())
                if path is not None:
                    images.append(Image.fromFileSystem(path))
                    self._since[umo] = 0
                    logger.info(f"[sticker] 附表情包 {tags[0]} -> {path.name}")
            else:
                self._since[umo] = since + 1
        elif since < cooldown:
            self._since[umo] = since + 1

        new_chain = [Plain(p) for p in parts] + others + images

        voice_text = "".join(parts).strip()
        vcd = int(CFG.get("voice_cooldown", 6))
        vmin = int(CFG.get("voice_min_chars", 4))
        vmax = int(CFG.get("voice_max_chars", 40))
        vsince = self._voice.get(umo, vcd)
        if vsince >= vcd and vmin <= len(voice_text) <= vmax and not images:
            audio = _synth(voice_text)
            if audio is not None:
                self._voice[umo] = 0
                new_chain = [Record.fromFileSystem(audio)]
            else:
                self._voice[umo] = vsince + 1
        else:
            self._voice[umo] = vsince + 1

        if not new_chain:
            return
        result.chain.clear()
        result.chain.extend(new_chain)

    @filter.on_llm_request()
    async def inject_sticker_rule(self, event: AstrMessageEvent, req):
        avail = "、".join(f"{x}({len(_images(x))})" for x in LABELS if _images(x))
        if not avail:
            return
        req.system_prompt = (req.system_prompt or "") + (
            "\n\n【表情包】每条回复末尾都要加一个情绪标记，格式 [[表情:开心]]；"
            "如果这条确实没有情绪，可以不加。"
            f"可用标签（括号内是库存）：{avail}。"
        )

    @filter.command("表情")
    async def send_sticker(self, event: AstrMessageEvent, tag: str = ""):
        tag = tag.strip() or "开心"
        path = _pick(tag)
        if path is None:
            yield event.plain_result("表情包库是空的")
            return
        yield event.image_result(str(path))

    @filter.command("表情列表")
    async def list_stickers(self, event: AstrMessageEvent):
        root = _sticker_root()
        if not root.is_dir():
            yield event.plain_result(f"表情包目录不存在：{root}")
            return
        lines = []
        for d in sorted(root.iterdir()):
            if d.is_dir():
                n = len(_images(d.name))
                if n:
                    lines.append(f"{d.name}: {n} 张")
        yield event.plain_result("表情包库存：\n" + "\n".join(lines) if lines else "表情包目录是空的")

    @filter.command("语音")
    async def say_voice(self, event: AstrMessageEvent, text: str = ""):
        text = text.strip()
        if not text:
            yield event.plain_result("用法：/语音 在呀鱼宝")
            return
        audio = _synth(text[: int(CFG.get("voice_max_chars", 40))])
        if audio is None:
            yield event.plain_result("语音合成失败，检查 tts_url 配置")
            return
        yield event.chain_result([Record.fromFileSystem(audio)])

    @filter.command("存表情")
    async def save_sticker(self, event: AstrMessageEvent, tag: str = ""):
        tag = tag.strip()
        if not tag:
            yield event.plain_result("用法：引用一张图，发 /存表情 开心")
            return
        images = [c for c in event.get_messages() if isinstance(c, Image)]
        if not images:
            yield event.plain_result("没找到图片，请引用一张图再发这条")
            return
        target = _sticker_root() / tag
        target.mkdir(parents=True, exist_ok=True)
        saved = 0
        for comp in images:
            try:
                src = await comp.convert_to_file_path()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[sticker] 取图失败: {exc}")
                continue
            src_path = Path(src)
            if not src_path.is_file():
                continue
            dst = target / f"{random.randint(100000, 999999)}{src_path.suffix.lower()}"
            shutil.copyfile(src_path, dst)
            saved += 1
        if saved:
            yield event.plain_result(f"存好了，{tag} 现在有 {len(_images(tag))} 张")
        else:
            yield event.plain_result("没存成功，可能是图片下载失败")
