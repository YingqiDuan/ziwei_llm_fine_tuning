# -*- coding: utf-8 -*-
"""
紫微斗数排盘（修订版，依据安星诀与节气法校正）

主要修订点：
1) 节气边界：改为“精确到时刻”的节气（UTC→本地时），杜绝仅按日期判断导致的跨日误判。
2) 月柱（节气月）与年柱（立春界）：以立春本地时刻为界，严格判年、判月。
3) 紫微锚定：依安星诀「六五四三二，酉午亥辰丑；局数除日数，商数宫前走；若见数无余，便要起虎口；日数小于局，还直宫中守」的“补数奇逆偶顺”算法实现（与 NCC/iztro 口径一致）。
4) 五行局数：修正“差值→五行”的映射（1木/2金/3水/4火/5土），示例与通行口诀一致。
5) 十二宫宫干：寅宫起干（五虎遁）→按地支顺序铺排到全盘，移除原先『亥宫起干+位移』的隐式偏移，以提升可读性与可验证性。
6) 太阳时/EoT：保留 NOAA 近似式，但结构化实现，代码可替换为更高精度天文库。

注：不同流派（南/北派、飞星、亮度表等）可能存在差异，本实现以常见口径为默认；关键表、规则均做成可替换。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
import csv
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

# ==========================
# 基础字典与工具
# ==========================
GAN = list("甲乙丙丁戊己庚辛壬癸")
ZHI = list("子丑寅卯辰巳午未申酉戌亥")
GAN_YANG = set("甲丙戊庚壬")
GAN_YIN  = set("乙丁己辛癸")
ZHI_IDX = {z:i for i,z in enumerate(ZHI)}
GAN_IDX = {g:i for i,g in enumerate(GAN)}

DATA_DIR = Path(__file__).parent / "data"

def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

palace_rows = _load_csv(DATA_DIR / "palace_order.csv")
PALACE_ORDER = [row['name'] for row in sorted(palace_rows, key=lambda r: int(r['index']))]

@dataclass
class BasicInfo:
    gender: str  # "男" or "女"
    longitude: float  # 地理经度（东经为正）
    clock_time: datetime  # 钟表时间（本地时区）
    true_solar_time: datetime  # 真太阳时
    lunar_text: Optional[str] = None
    pillars_qi: Optional[Tuple[str,str,str,str]] = None  # 节气四柱 干支
    pillars_non_qi: Optional[Tuple[str,str,str,str]] = None  # 非节气四柱 干支
    wuxing_ju: Optional[str] = None  # 如 "火六局"
    shen_zhu: Optional[str] = None
    ming_zhu: Optional[str] = None
    shen_gong_zhi: Optional[str] = None  # 身宫地支

@dataclass
class StarPlacement:
    main: List[str] = field(default_factory=list)   # 主星
    assist: List[str] = field(default_factory=list) # 辅星
    brightness: Dict[str, str] = field(default_factory=dict)  # 庙旺利陷 等
    transforms: Dict[str, List[str]] = field(default_factory=dict)  # 生年禄/权/科/忌 标注

@dataclass
class Palace:
    name: str
    gan: str
    zhi: str
    tags: List[str] = field(default_factory=list)  # 如 ["来因"、"身宫"]
    placement: StarPlacement = field(default_factory=StarPlacement)
    daxian: Tuple[int,int] = (0,0)  # 起止虚岁
    liunian_list: List[int] = field(default_factory=list)

@dataclass
class Chart:
    basic: BasicInfo
    palaces: List[Palace]

# ==========================
# 天文：真太阳时（NOAA 近似式）
# ==========================

def _day_of_year(dt_utc: datetime) -> float:
    start = datetime(dt_utc.year, 1, 1, tzinfo=timezone.utc)
    return (dt_utc - start).total_seconds() / 86400.0 + 1.0

def equation_of_time_minutes(dt_utc: datetime) -> float:
    """均时差（分钟）。NOAA 近似式，精度达分钟级，足够术数用途。
    参考：NOAA Solar Calculator 公式。
    """
    doy = _day_of_year(dt_utc)
    gamma = 2*math.pi/365.0 * (doy - 1 + (dt_utc.hour - 12)/24.0)
    eq = (229.18 * (
        0.000075 + 0.001868*math.cos(gamma) - 0.032077*math.sin(gamma)
        - 0.014615*math.cos(2*gamma) - 0.040849*math.sin(2*gamma)
    ))
    return eq  # minutes

def true_solar_time(local: datetime, longitude_deg: float, tz_offset_hours: float) -> datetime:
    """时区时 → 真太阳时。
    local: 本地时区钟表时（若 naive，则按 tz_offset_hours 解释）
    """
    if local.tzinfo is None:
        tz = timezone(timedelta(hours=tz_offset_hours))
        local = local.replace(tzinfo=tz)
    dt_utc = local.astimezone(timezone.utc)
    lstm = 15.0 * tz_offset_hours  # 标准子午线
    tc_minutes = 4.0*(longitude_deg - lstm) + equation_of_time_minutes(dt_utc)
    return local + timedelta(minutes=tc_minutes)

# ==========================
# 干支工具（年/月/日/时）
# ==========================

def ganzhi_from_index(idx60: int) -> str:
    return GAN[idx60%10] + ZHI[idx60%12]

# 五虎遁：年干→寅月干
FIVE_TIGER = {row['year_gan']: row['start_month_gan'] for row in _load_csv(DATA_DIR / "five_tiger.csv")}

# 五鼠遁：日干→子时干
FIVE_MOUSE = {row['day_gan']: row['zi_hour_gan'] for row in _load_csv(DATA_DIR / "five_mouse.csv")}

# 时支映射（真太阳时）
HOUR_BRANCHES = [
    (row['branch'], int(row['start_hour']), int(row['end_hour']))
    for row in _load_csv(DATA_DIR / "hour_branches.csv")
]

def branch_by_true_time(ts: datetime) -> str:
    h = ts.hour + ts.minute/60.0
    for z, start, end in HOUR_BRANCHES:
        if z=='子':
            if h>=23 or h<1:
                return '子'
        else:
            if start <= h < end:
                return z
    return '子'

# ============== 节气（常用 sTermInfo 近似） ==============
# 以 1900-01-06 02:05:00 UTC 为基点，月份节(0..23)的毫秒偏移常数表。
S_TERM_INFO = [
    0,21208,42467,63836,85337,107014,128867,150921,173149,195551,218072,240693,
    263343,285989,308563,331033,353350,375494,397447,419210,440795,462224,483532,504758
]
EPOCH_1900 = datetime(1900,1,6,2,5, tzinfo=timezone.utc)
TROPICAL_YEAR_MS = 31556925974.7

def solar_term_dt_utc(year:int, n:int) -> datetime:
    """返回该年第 n 个节气（0-23）的 *UTC* 发生时刻（近似）。"""
    ms = TROPICAL_YEAR_MS*(year-1900) + S_TERM_INFO[n]*60000
    return EPOCH_1900 + timedelta(milliseconds=ms)

def solar_term_dt_local(year:int, n:int, tz_offset_hours: float) -> datetime:
    dt_utc = solar_term_dt_utc(year, n)
    tz = timezone(timedelta(hours=tz_offset_hours))
    return dt_utc.astimezone(tz)

# 干支纪年（节气法：以立春为界，立春为 index=2）

def ganzhi_year_by_jieqi(dt_local: datetime, tz_offset_hours: float) -> str:
    y = dt_local.year
    li_chun = solar_term_dt_local(y, 2, tz_offset_hours)
    if dt_local < li_chun:
        y -= 1
    # 1984 甲子为周期零点
    offset = (y - 1984) % 60
    return ganzhi_from_index(offset)

# 干支纪月（节气月：每月在“节”切换；寅月自立春始）
JIE_INDEX_BY_MONTH = [
    0,  # placeholder
    0,  # Jan: 小寒(0)
    2,  # Feb: 立春(2)
    4,  # Mar: 惊蛰(4)
    6,  # Apr: 清明(6)
    8,  # May: 立夏(8)
    10, # Jun: 芒种(10)
    12, # Jul: 小暑(12)
    14, # Aug: 立秋(14)
    16, # Sep: 白露(16)
    18, # Oct: 寒露(18)
    20, # Nov: 立冬(20)
    22, # Dec: 大雪(22)
]
BRANCH_BY_QI_MONTH = ['寅','卯','辰','巳','午','未','申','酉','戌','亥','子','丑']

FIVE_TIGER_GROUP = FIVE_TIGER  # alias

def month_ganzhi_by_jieqi(dt_local: datetime, year_gan: str, tz_offset_hours: float) -> str:
    # 判断当月是否已过“节”
    idx = JIE_INDEX_BY_MONTH[dt_local.month]
    term_local = solar_term_dt_local(dt_local.year, idx, tz_offset_hours)
    month_index = (dt_local.month - 2) % 12
    if dt_local < term_local:
        month_index = (month_index - 1) % 12
    zhi = BRANCH_BY_QI_MONTH[month_index]
    gan0 = FIVE_TIGER_GROUP[year_gan]
    gan = GAN[(GAN_IDX[gan0] + month_index) % 10]
    return gan + zhi

# 干支纪日（以 1912-02-18 为甲子日，JDN 基准映射）
# 参照通行算法：JDN 整日对应本地日，offset 使 1912-02-18 → 甲子。

def ganzhi_day(dt_local: datetime) -> str:
    year, month, day = dt_local.year, dt_local.month, dt_local.day
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    jdn = (int(365.25 * (year + 4716))
           + int(30.6001 * (month + 1))
           + day + B - 1524)
    # 1912-02-18 的 JDN 设为 甲子（idx=0）
    JDN_1912_02_18 = 2419451
    idx = (jdn - JDN_1912_02_18) % 60
    return ganzhi_from_index(idx)

# 干支纪时：五鼠遁 + 时支

def ganzhi_hour(day_gan: str, hour_zhi: str) -> str:
    gan_at_zi = FIVE_MOUSE[day_gan]
    dz = ZHI_IDX[hour_zhi]
    gan = GAN[(GAN_IDX[gan_at_zi] + dz) % 10]
    return gan + hour_zhi

from lunardate import LunarDate  # pip install lunardate

@dataclass
class LunarInfo:
    year_gz: Optional[str]
    month: Optional[int]
    day: Optional[int]
    is_leap: Optional[bool]
    text: Optional[str]

LUNAR_MONTH_NAME = ["正","二","三","四","五","六","七","八","九","十","冬","腊"]
LUNAR_DAY_NAME = [
    "初一","初二","初三","初四","初五","初六","初七","初八","初九","初十",
    "十一","十二","十三","十四","十五","十六","十七","十八","十九","二十",
    "廿一","廿二","廿三","廿四","廿五","廿六","廿七","廿八","廿九","三十"
]

# ============== 五行局 ==============
# 干支取数后相加，>5 减 5；差值→五行：1木/2金/3水/4火/5土
WUXING_NAME = {int(row['number']): row['label'] for row in _load_csv(DATA_DIR / "wuxing_name.csv")}
GAN_TO_NUM = {row['gan']: int(row['value']) for row in _load_csv(DATA_DIR / "gan_to_num.csv")}
ZHI_TO_NUM = {row['zhi']: int(row['value']) for row in _load_csv(DATA_DIR / "zhi_to_num.csv")}

# 将“五行局名字”转为数字（2/3/4/5/6）
def ju_number(ju_name: str) -> int:
    mapping = {"二":2, "三":3, "四":4, "五":5, "六":6}
    for ch, num in mapping.items():
        if ch in ju_name:
            return num
    raise ValueError(f"无法识别五行局数: {ju_name}")

# ============== 四化（南派口径） ==============
SIHUA: Dict[str, Dict[str, str]] = {}
for row in _load_csv(DATA_DIR / "sihua.csv"):
    SIHUA.setdefault(row['year_gan'], {})[row['role']] = row['star']

# ============== 重要辅曜映射 ==============
LUCUN_BY_GAN = {row['gan']: row['zhi'] for row in _load_csv(DATA_DIR / "lucun_by_gan.csv")}
KUIYUE_BY_GAN = {
    row['gan']: (row['tian_kui_zhi'], row['tian_yue_zhi'])
    for row in _load_csv(DATA_DIR / "kuiyue_by_gan.csv")
}
TIANMA_BY_ZHI_GROUP = {row['group']: row['zhi'] for row in _load_csv(DATA_DIR / "tianma_by_zhi_group.csv")}
HUOLING_GROUP = {
    row['group']: (row['huo_start'], row['ling_start'])
    for row in _load_csv(DATA_DIR / "huoling_group.csv")
}

# ============== 亮度（庙旺…） ==============
BRIGHTNESS_TABLE: Dict[str, Dict[str, str]] = {}
for row in _load_csv(DATA_DIR / "brightness_table.csv"):
    BRIGHTNESS_TABLE.setdefault(row['star'], {})[row['zhi']] = row['status']

# ============== 命主/身主（常见口径存在差异，可按需替换） ==============
MINGZHU_BY_PALACE_ZHI = {row['zhi']: row['star'] for row in _load_csv(DATA_DIR / "mingzhu_by_palace_zhi.csv")}
SHENZHU_BY_YEAR_ZHI = {row['year_zhi']: row['star'] for row in _load_csv(DATA_DIR / "shenzhu_by_year_zhi.csv")}

# ==========================
# 宫位天干：寅位起干，顺排到全盘
# ==========================

def palace_gans_by_year_gan(year_gan: str) -> List[str]:
    gans = [None]*12
    gan_at_yin = FIVE_TIGER[year_gan]
    start = ZHI_IDX['寅']
    for i in range(12):
        z_idx = (start + i) % 12
        gans[z_idx] = GAN[(GAN_IDX[gan_at_yin] + i) % 10]
    return gans  # 与 ZHI 顺序对齐（子..亥）

# ==========================
# 命宫/身宫
# ==========================

def palace_index_of_ming(lunar_month:int, hour_branch:str) -> int:
    base = ZHI_IDX['寅']
    p = (base + (lunar_month-1)) % 12
    h = ZHI_IDX[hour_branch]
    p = (p - h) % 12
    return p

def palace_index_of_shen(lunar_month:int, hour_branch:str) -> int:
    base = ZHI_IDX['寅']
    p = (base + (lunar_month-1)) % 12
    h = ZHI_IDX[hour_branch]
    p = (p + h) % 12
    return p

# ==========================
# 五行局数（以『命宫干支』为准）
# ==========================

def wuxing_ju_by_ming_ganzhi(ming_gan: str, ming_zhi: str) -> str:
    """以『命宫干支』定五行局（干支取数相加，>5 减 5）：
    甲乙=1、丙丁=2、戊己=3、庚辛=4、壬癸=5；
    子午丑未=1、寅申卯酉=2、辰戌巳亥=3。
    1→木三局、2→金四局、3→水二局、4→火六局、5→土五局。
    参照安星诀『干支相加多减五，五行木金水火土』。
    """
    x = GAN_TO_NUM[ming_gan] + ZHI_TO_NUM[ming_zhi]
    while x > 5:
        x -= 5
    return WUXING_NAME[x]

# ==========================
# 紫微起星（安星诀：补数奇逆、偶顺；虎口=寅）
# ==========================

def ziwei_anchor_by_day_and_ju(lunar_day:int, ju:int) -> int:
    # s = ceil(day/ju)；补数 x = s*ju - day
    s = (lunar_day + ju - 1) // ju
    x = s*ju - lunar_day
    start = (ZHI_IDX['寅'] + (s - 1)) % 12  # 从寅起进(商-1)格
    if x == 0:
        return start  # 整除，停在起点（例：27日、木三局 → 戌）
    if x % 2 == 1:
        return (start - x) % 12  # 补数奇：逆回 x 宫
    else:
        return (start + x) % 12  # 补数偶：顺行 x 宫

ZIWEI_SERIES = [(row['star'], int(row['offset'])) for row in _load_csv(DATA_DIR / "ziwei_series.csv")]

# 天府：由紫微定位，按“寅申中轴”映射
ZIWEI_TO_TIANFU = {row['ziwei_zhi']: row['tianfu_zhi'] for row in _load_csv(DATA_DIR / "ziwei_to_tianfu.csv")}

TIANFU_SERIES = [(row['star'], int(row['offset'])) for row in _load_csv(DATA_DIR / "tianfu_series.csv")]

# ==========================
# 口诀型杂曜安置
# ==========================

def place_left_right_by_month(lunar_month:int) -> Tuple[str,str]:
    left = (ZHI_IDX['辰'] + (lunar_month-1)) % 12
    right = (ZHI_IDX['戌'] - (lunar_month-1)) % 12
    return ZHI[left], ZHI[right]

def place_wenchang_wenqu_by_hour(hour_branch:str) -> Tuple[str,str]:
    dz = ZHI_IDX[hour_branch]
    chang = (ZHI_IDX['戌'] - dz) % 12
    qu    = (ZHI_IDX['辰'] + dz) % 12
    return ZHI[chang], ZHI[qu]

def daxian_ranges(gender:str, year_gan:str, start_palace:int, ju:int) -> List[Tuple[int,Tuple[int,int]]]:
    start_age = ju if ju >= 2 else 6
    forward = ((gender=='男' and year_gan in GAN_YANG) or (gender=='女' and year_gan in GAN_YIN))
    seq = [start_palace]
    for _ in range(1,12):
        seq.append((seq[-1] + (1 if forward else -1)) % 12)
    age0 = start_age
    return [(p, (age0 + i*10, age0 + i*10 + 9)) for i, p in enumerate(seq)]

def liunian_for_palace(birth_year_zhi:str, palace_zhi:str, up_to:int=60) -> List[int]:
    b = ZHI_IDX[birth_year_zhi]
    p = ZHI_IDX[palace_zhi]
    return [age for age in range(1, up_to+1) if (b + (age-1)) % 12 == p]

# ==========================
# 生成命盘
# ==========================

@dataclass
class InputData:
    gender: str  # "男"/"女"
    longitude: float
    tz: float  # 时区小时
    dt_clock: datetime  # 钟表时（本地时）


def build_chart(inp: InputData) -> Chart:
    # 统一：给 dt_clock 补上 tzinfo，供后续节气/四柱使用
    if inp.dt_clock.tzinfo is None:
        tz = timezone(timedelta(hours=inp.tz))
        dt_local = inp.dt_clock.replace(tzinfo=tz)
    else:
        dt_local = inp.dt_clock

    # 1) 真太阳时
    ts = true_solar_time(dt_local, inp.longitude, inp.tz)
    hour_branch = branch_by_true_time(ts)

    # 2) 节气四柱（以本地“节气时刻”为界）
    y_gz_qi = ganzhi_year_by_jieqi(dt_local, inp.tz)
    m_gz_qi = month_ganzhi_by_jieqi(dt_local, y_gz_qi[0], inp.tz)
    d_gz = ganzhi_day(dt_local)
    h_gz = ganzhi_hour(d_gz[0], hour_branch)

    # 3) 农历信息（使用 lunardate）
    lunar_obj = LunarDate.fromSolarDate(dt_local.year, dt_local.month, dt_local.day)
    lunar = LunarInfo(
        year_gz=ganzhi_from_index((lunar_obj.year - 1984) % 60),
        month=lunar_obj.month,
        day=lunar_obj.day,
        is_leap=bool(lunar_obj.isLeapMonth),
        text=None
    )
    lunar.text = f"{lunar.year_gz}年{('闰' if lunar.is_leap else '')}{LUNAR_MONTH_NAME[lunar.month-1]}月{LUNAR_DAY_NAME[lunar.day-1]}{hour_branch}时"

    # 规则使用的年干支 = 农历年干支
    y_gz_lunar = lunar.year_gz
    year_gan_rules = y_gz_lunar[0]
    year_zhi_rules = y_gz_lunar[1]

    # 4) 命宫/身宫
    lm = lunar.month
    ld_day = lunar.day
    ming_idx = palace_index_of_ming(lm, hour_branch)
    shen_idx = palace_index_of_shen(lm, hour_branch)
    shen_zhi = ZHI[shen_idx]

    # 5) 宫干与十二宫名（寅起干 → 顺排到全盘）
    gans = palace_gans_by_year_gan(year_gan_rules)
    palaces: List[Palace] = [
        Palace(
            name=PALACE_ORDER[(ming_idx - i) % 12],
            gan=gans[i],  # 直接按地支子..亥取宫干
            zhi=ZHI[i],
            tags=['身宫'] if i == shen_idx else []
        )
        for i in range(12)
    ]

    # 6) 五行局（以命宫干支：命宫天干 + 命宫地支）
    ming_gan = palaces[ming_idx].gan
    ju_name = wuxing_ju_by_ming_ganzhi(ming_gan, ZHI[ming_idx])
    ju_num = ju_number(ju_name)

    # 7) 十四主星安置
    ziwei_pos = ziwei_anchor_by_day_and_ju(ld_day, ju_num)
    for star, off in ZIWEI_SERIES:
        pos = (ziwei_pos + off) % 12
        palaces[pos].placement.main.append(star)
    tf_zhi = ZIWEI_TO_TIANFU[ZHI[ziwei_pos]]
    tianfu_pos = ZHI_IDX[tf_zhi]
    for star, off in TIANFU_SERIES:
        pos = (tianfu_pos + off) % 12
        palaces[pos].placement.main.append(star)

    # 8) 生年四化标注
    sihua = SIHUA[year_gan_rules]
    for k, star in sihua.items():
        for p in palaces:
            if star in p.placement.main or star in p.placement.assist:
                p.placement.transforms.setdefault(star,[]).append(f"生年{k}")

    # 9) 辅曜
    lucun_zhi = LUCUN_BY_GAN[year_gan_rules]
    lucun_pos = ZHI_IDX[lucun_zhi]
    palaces[lucun_pos].placement.assist.append('禄存')
    yang_pos = (lucun_pos - 1) % 12
    tuo_pos  = (lucun_pos + 1) % 12
    palaces[yang_pos].placement.assist.append('擎羊')
    palaces[tuo_pos ].placement.assist.append('陀罗')

    zf_zhi, yb_zhi = place_left_right_by_month(lm)
    palaces[ZHI_IDX[zf_zhi]].placement.assist.append('左辅')
    palaces[ZHI_IDX[yb_zhi]].placement.assist.append('右弼')

    wc_zhi, wq_zhi = place_wenchang_wenqu_by_hour(hour_branch)
    palaces[ZHI_IDX[wc_zhi]].placement.assist.append('文昌')
    palaces[ZHI_IDX[wq_zhi]].placement.assist.append('文曲')

    k, y_ = KUIYUE_BY_GAN[year_gan_rules]
    palaces[ZHI_IDX[k]].placement.assist.append('天魁')
    palaces[ZHI_IDX[y_]].placement.assist.append('天钺')

    tianma_zhi = next((z for grp, z in TIANMA_BY_ZHI_GROUP.items() if year_zhi_rules in grp), None)
    if tianma_zhi:
        palaces[ZHI_IDX[tianma_zhi]].placement.assist.append('天马')

    for grp, (huo_start, ling_start) in HUOLING_GROUP.items():
        if year_zhi_rules in grp:
            dz = ZHI_IDX[hour_branch]
            huo_pos = (ZHI_IDX[huo_start] + dz) % 12
            ling_pos = (ZHI_IDX[ling_start] + dz) % 12
            palaces[huo_pos].placement.assist.append('火星')
            palaces[ling_pos].placement.assist.append('铃星')
            break

    # 10) 亮度标注
    for p in palaces:
        for star in p.placement.main + p.placement.assist:
            tab = BRIGHTNESS_TABLE.get(star)
            if tab:
                br = tab.get(p.zhi)
                if br:
                    p.placement.brightness[star] = br

    # 11) 大限
    dx = daxian_ranges(inp.gender, y_gz_qi[0], ming_idx, ju_num)
    for pidx, (a,b) in dx:
        palaces[pidx].daxian = (a,b)

    # 12) 流年（1..60）
    for p in palaces:
        p.liunian_list  = liunian_for_palace(year_zhi_rules, p.zhi, 60)

    # 13) 命主/身主
    ming_palace_zhi = palaces[ming_idx].zhi
    ming_zhu = MINGZHU_BY_PALACE_ZHI.get(ming_palace_zhi)
    year_zhi_for_shenzhu = (lunar.year_gz or y_gz_lunar)[1]  # 年柱地支（非节气年）
    shen_zhu = SHENZHU_BY_YEAR_ZHI[year_zhi_for_shenzhu]

    basic = BasicInfo(
        gender=inp.gender,
        longitude=inp.longitude,
        clock_time=dt_local,
        true_solar_time=ts,
        lunar_text=lunar.text,
        pillars_qi=(y_gz_qi, m_gz_qi, d_gz, h_gz),
        pillars_non_qi=None,
        wuxing_ju=ju_name,
        shen_zhu=shen_zhu,
        ming_zhu=ming_zhu,
        shen_gong_zhi=shen_zhi
    )

    ygz = y_gz_lunar
    yg = ygz[0]
    yz_branch = ZHI[(ZHI_IDX['寅'] + (lm - 1)) % 12]
    gan0 = FIVE_TIGER[yg]
    mg = GAN[(GAN_IDX[gan0] + (lm - 1)) % 10]
    basic.pillars_non_qi = (ygz, mg + yz_branch, d_gz, h_gz)

    return Chart(basic=basic, palaces=palaces)

# ==========================
# 文本输出
# ==========================

def fmt_range(a,b):
    return f"{a}~{b}虚岁"

def render_chart(chart: Chart) -> str:
    b = chart.basic
    lines = []

    def format_star_line(label: str, stars: List[str], placement: StarPlacement, include_transforms: bool = False) -> str:
        if not stars:
            return f"│ │ ├{label} : 无"
        formatted = []
        for st in stars:
            text = st
            br = placement.brightness.get(st)
            if br:
                text += f"[{br}]"
            if include_transforms:
                for tag in placement.transforms.get(st, []):
                    text += f"[{tag}]"
            formatted.append(text)
        return f"│ │ ├{label} : " + ",".join(formatted)

    lines.append("紫微斗数命盘")
    lines.append("│")
    lines.append("├符号定义 : (↓:离心自化), (↑:向心自化，从对宫化入)")
    lines.append("│")
    lines.append("├基本信息")
    lines.append("│ │")
    lines.append(f"│ ├性别 : {b.gender}")
    lines.append(f"│ ├地理经度 : {b.longitude}")
    lines.append(f"│ ├钟表时间 : {b.clock_time.strftime('%Y-%-m-%-d %H:%M') if hasattr(b.clock_time,'strftime') else b.clock_time}")
    lines.append(f"│ ├真太阳时 : {b.true_solar_time.strftime('%Y-%-m-%-d %H:%M')}")
    if b.lunar_text:
        lines.append(f"│ ├农历时间 : {b.lunar_text}")
    if b.pillars_qi:
        y,m,d,h = b.pillars_qi
        lines.append(f"│ ├节气四柱 : {y} {m} {d} {h}")
    if b.pillars_non_qi:
        y,m,d,h = b.pillars_non_qi
        lines.append(f"│ ├非节气四柱 : {y} {m} {d} {h}")
    if b.wuxing_ju:
        lines.append(f"│ ├五行局数 : {b.wuxing_ju}")
    tag = []
    if b.shen_zhu: tag.append(f"身主:{b.shen_zhu}")
    if b.ming_zhu: tag.append(f"命主:{b.ming_zhu}")
    if b.shen_gong_zhi:
        tag.append(f"身宫:{b.shen_gong_zhi}")
    if tag:
        lines.append("│ └" + "; ".join(tag))
    lines.append("│")
    lines.append("├命盘十二宫")
    lines.append("│ │ ")

    ming_idx = next(i for i, p in enumerate(chart.palaces) if p.name == '命宫')
    order = [(ming_idx + i) % 12 for i in range(12)]
    for idx in order:
        p = chart.palaces[idx]
        head = f"│ ├{p.name}[{p.gan}{p.zhi}]"
        if p.tags:
            head += "["+"、".join(p.tags)+"]"
        lines.append(head)
        lines.append(format_star_line("主星", p.placement.main, p.placement, include_transforms=True))
        lines.append(format_star_line("辅星", p.placement.assist, p.placement))
        a,b_ = p.daxian
        lines.append(f"│ │ ├大限 : {fmt_range(a,b_)}")
        if p.liunian_list:
            lines.append("│ │ └流年 : " + ",".join(str(x) for x in p.liunian_list[:5]) + "虚岁")
        lines.append("│ │ ")

    lines.append("│")
    lines.append("└")
    return "\n".join(lines)

# ==========================
# 使用示例
# ==========================
if __name__ == "__main__":
    # 示例：2002-03-08 17:50（UTC+8），经度 112.733，男
    dt = datetime(2002,3,8,17,50)
    chart = build_chart(InputData(
        gender='男',
        longitude=112.733,
        tz=8,
        dt_clock=dt
    ))
    print(render_chart(chart))
