# -*- coding: utf-8 -*-
"""
紫微斗数排盘（可运行、可扩展）

功能覆盖：
1) 真太阳时校正（基于经度与均时差 EoT）
2) 四柱（节气法/非节气法）与干支计算（五虎遁、五鼠遁），可选用节气法
3) 命宫、身宫、十二宫干支与顺序（定寅首、定十二宫）
4) 五行局数（含名称）
5) 十四主星安置：
   - 紫微系：紫微→天机→(空一)→太阳→武曲→天同→(空二)→廉贞（逆行）
   - 天府系：天府→太阴→贪狼→巨门→天相→天梁→七杀→(空三)→破军（顺行）
   - 天府起点由紫微定位按“寅申中轴”映射
6) 生年四化（南派四化）映射：禄/权/科/忌
7) 重要辅曜：
   - 禄存、擎羊、陀罗（按年干；羊逆两位、陀顺两位）
   - 文昌、文曲（戌上起子逆/辰上起子顺，按时支）
   - 左辅、右弼（辰上起正顺/戌上起正逆，按月）
   - 天魁、天钺（按年干）
   - 天马（按年支四组）
   - 火星、铃星（按年支四组+时支）
8) 大限/流年落宫：
    - 大限：由命宫起，每宫十年；起始虚岁=局数；阳男阴女顺、阴男阳女逆
    - 流年：以出生年支递进（按岁数）落在同名地支之宫；输出 1..60 岁(虚岁)落点
9) 星曜庙旺利陷：内置常用表（主星+昌曲辅弼等）

依赖：
- 标配：仅用 Python 标准库
- 可选：lunardate (用于公历↔农历转换)。若未安装，将退化为：
  • 仍可排：节气四柱、命盘宫位、真太阳时、十四主星（需农历日→若无则需手工传入 lunar_day）
  • 为保证通用性，你可安装：pip install lunardate

参考实现/校验资料（口诀与对照）：
- 定寅首、安命身宫、起大限、安紫微/天府诸星、四化、辅弼昌曲、魁钺、禄存羊陀、天马、火铃等传统口诀。
- 亮度（庙旺失陷）表的通行版本。

—— 注意 ——
各派有差异（南北派、飞星体系、亮度口径等）。本脚本按常见口径实现，并尽量模块化，便于你替换表与规则。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
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

PALACE_ORDER = [
    "命宫","兄弟宫","夫妻宫","子女宫","财帛宫","疾厄宫",
    "迁移宫","交友宫","官禄宫","田宅宫","福德宫","父母宫"
]

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
# 天文：真太阳时（简式）
# ==========================

def equation_of_time_minutes(dt_utc: datetime) -> float:
    """均时差（分钟），近似公式（精度±1分量级即可满足术数用途）"""
    # 参考 NOAA 近似式
    # B = 2π*(N-81)/364, N 年内序号
    start_year = datetime(dt_utc.year, 1, 1, tzinfo=timezone.utc)
    N = (dt_utc - start_year).days + 1
    B = 2*math.pi*(N-81)/364.0
    E = 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)
    return E  # minutes

def true_solar_time(local: datetime, longitude_deg: float, tz_offset_hours: float) -> datetime:
    """由时区时→真太阳时
    local: 本地时区钟表时（带 tzinfo 或假定 tz_offset_hours）
    tz_offset_hours: 时区东八区=8
    """
    if local.tzinfo is None:
        tz = timezone(timedelta(hours=tz_offset_hours))
        local = local.replace(tzinfo=tz)
    dt_utc = local.astimezone(timezone.utc)
    LSTM = 15.0*tz_offset_hours  # 标准子午线
    tc_minutes = 4.0*(longitude_deg - LSTM) + equation_of_time_minutes(dt_utc)
    return local + timedelta(minutes=tc_minutes)

# ==========================
# 干支工具（年/月/日/时）
# ==========================

def ganzhi_from_index(idx60: int) -> str:
    return GAN[idx60%10] + ZHI[idx60%12]

# 五虎遁：年干→寅月干
FIVE_TIGER = {
    '甲':'丙','己':'丙',
    '乙':'戊','庚':'戊',
    '丙':'庚','辛':'庚',
    '丁':'壬','壬':'壬',
    '戊':'甲','癸':'甲',
}

# 五鼠遁：日干→子时干
FIVE_MOUSE = {
    '甲':'甲','己':'甲',
    '乙':'丙','庚':'丙',
    '丙':'戊','辛':'戊',
    '丁':'庚','壬':'庚',
    '戊':'壬','癸':'壬',
}

# 时支映射
HOUR_BRANCHES = [
    ('子',23,1),('丑',1,3),('寅',3,5),('卯',5,7),('辰',7,9),('巳',9,11),
    ('午',11,13),('未',13,15),('申',15,17),('酉',17,19),('戌',19,21),('亥',21,23)
]

def branch_by_true_time(ts: datetime) -> str:
    h = ts.hour + ts.minute/60.0
    for z, start, end in HOUR_BRANCHES:
        # 子时跨日，特判
        if z=='子':
            if h>=23 or h<1:
                return '子'
        else:
            if start <= h < end:
                return z
    return '子'

# ============== 节气（朔望历近似） ==============
# 采用常见 sTermInfo 常数以天文近似法计算节气日（UTC 基准），用于“节气四柱”的年/月界定。
# 常数与方法来源：常见万年历实现。

S_TERM_INFO = [
    0,21208,42467,63836,85337,107014,128867,150921,173149,195551,218072,240693,
    263343,285989,308563,331033,353350,375494,397447,419210,440795,462224,483532,504758
]

EPOCH_1900 = datetime(1900,1,6,2,5, tzinfo=timezone.utc)  # 常用基点
TROPICAL_YEAR_MS = 31556925974.7

def get_solar_term_day_utc(year: int, n: int) -> int:
    # 返回公历该年第 n 个节气（0-23）发生的“日”（UTC 下在该月的日数）
    ms = TROPICAL_YEAR_MS*(year-1900) + S_TERM_INFO[n]*60000
    dt = EPOCH_1900 + timedelta(milliseconds=ms)
    return dt.day

# 干支纪年（节气法：以立春为界）
JIA_ZI_START_1984 = 60  # 1984 甲子年在 60 周期边界处（设定索引用，不影响结果相对差）

def ganzhi_year_by_jieqi(dt_local: datetime) -> str:
    y = dt_local.year
    # 立春通常为节气索引 2（小寒=0, 大寒=1, 立春=2...）
    li_chun_day = get_solar_term_day_utc(y, 2)
    y_adj = y
    if dt_local.month==2 and dt_local.day < li_chun_day:
        y_adj -= 1
    elif dt_local.month < 2:
        y_adj -= 1
    # 1984 甲子
    offset = (y_adj - 1984) % 60
    return ganzhi_from_index((0 + offset) % 60)  # 1984 甲子设为 idx0

def ganzhi_year_by_lunar_year(lunar_year: int) -> str:
    # 1984 为甲子年
    offset = (lunar_year - 1984) % 60
    return ganzhi_from_index(offset)


# 干支纪月（节气月：每月在“节”切换）
# 寅月自立春始，往后顺序卯月=惊蛰，辰=清明...

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

FIVE_TIGER_GROUP = {
    '甲': '丙', '己': '丙',
    '乙': '戊', '庚': '戊',
    '丙': '庚', '辛': '庚',
    '丁': '壬', '壬': '壬',
    '戊': '甲', '癸': '甲'
}

def month_ganzhi_by_jieqi(dt_local: datetime, year_gan: str) -> str:
    # 计算当月是否过“节”（非中气），若未过，则用上个月干支
    idx = JIE_INDEX_BY_MONTH[dt_local.month]
    term_day = get_solar_term_day_utc(dt_local.year, idx)
    # 月支序：立春=寅，惊蛰=卯...
    month_index = (dt_local.month - 2) % 12  # 粗略，以 2 月为寅起
    # 修正是否过当月之“节”
    if dt_local.day < term_day:
        month_index = (month_index - 1) % 12
    zhi = BRANCH_BY_QI_MONTH[month_index]
    # 月干由五虎遁
    gan0 = FIVE_TIGER[year_gan]
    gan = GAN[(GAN_IDX[gan0] + month_index) % 10]
    return gan + zhi

# 干支纪日：使用通用基准换算（简式）
# 以 1984-01-01 为甲子日（可换其他基点，保持相对差 60 周期）

def ganzhi_day(dt_local: datetime) -> str:
    # 将日期转为公历日序（不含时）
    base = datetime(1984,1,1, tzinfo=dt_local.tzinfo)
    days = (dt_local.date() - base.date()).days
    return ganzhi_from_index(days % 60)

# 干支纪时：五鼠遁 + 时支

def ganzhi_hour(day_gan: str, hour_zhi: str) -> str:
    gan_at_zi = FIVE_MOUSE[day_gan]
    dz = ZHI_IDX[hour_zhi]
    gan = GAN[(GAN_IDX[gan_at_zi] + dz) % 10]
    return gan + hour_zhi

from lunardate import LunarDate  # pip install lunardate
HAVE_LUNARDATE = True


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
# 按“干支取数相加，>5减5”定五行；并映射为 木三局/金四局/水二局/火六局/土五局
WUXING_NAME = {1:"木三局",2:"金四局",3:"水二局",4:"火六局",5:"土五局"}
GAN_TO_NUM = {"甲":1,"乙":1,"丙":2,"丁":2,"戊":3,"己":3,"庚":4,"辛":4,"壬":5,"癸":5}
ZHI_TO_NUM = {"子":1,"午":1,"丑":1,"未":1,"寅":2,"申":2,"卯":2,"酉":2,"辰":3,"戌":3,"巳":3,"亥":3}

# 将“五行局名字”转为数字（2/3/4/5/6）
def ju_number(ju_name: str) -> int:
    mapping = {"二":2, "三":3, "四":4, "五":5, "六":6}
    for ch, num in mapping.items():
        if ch in ju_name:
            return num
    raise ValueError(f"无法识别五行局数: {ju_name}")

# ============== 四化（南派口径） ==============
# 甲廉破武阳，乙机梁紫阴，丙同机昌廉，丁阴同机巨，戊贪阴右机。
# 己武贪梁曲，庚阳武阴同，辛巨阳曲昌，壬梁紫左武，癸破巨阴贪。
SIHUA = {
    '甲': {'禄':'廉贞','权':'破军','科':'武曲','忌':'太阳'},
    '乙': {'禄':'天机','权':'天梁','科':'紫微','忌':'太阴'},
    '丙': {'禄':'天同','权':'天机','科':'文昌','忌':'廉贞'},
    '丁': {'禄':'太阴','权':'天同','科':'天机','忌':'巨门'},
    '戊': {'禄':'贪狼','权':'太阴','科':'右弼','忌':'天机'},
    '己': {'禄':'武曲','权':'贪狼','科':'天梁','忌':'文曲'},
    '庚': {'禄':'太阳','权':'武曲','科':'太阴','忌':'天同'},
    '辛': {'禄':'文曲','权':'文昌','科':'廉贞','忌':'破军'},
    '壬': {'禄':'天梁','权':'紫微','科':'左辅','忌':'武曲'},
    '癸': {'禄':'破军','权':'太阴','科':'右弼','忌':'贪狼'},
}

# ============== 重要辅曜映射 ==============
# 禄存 by 年干
LUCUN_BY_GAN = {
    '甲':'寅','乙':'卯','丙':'巳','丁':'午','戊':'巳','己':'午','庚':'申','辛':'酉','壬':'亥','癸':'子'
}

# 魁钺
KUIYUE_BY_GAN = {
    '甲':('丑','未'),'戊':('丑','未'),'庚':('丑','未'),
    '乙':('子','申'),'己':('子','申'),
    '辛':('午','寅'),
    '壬':('卯','巳'),'癸':('卯','巳'),
    '丙':('亥','酉'),'丁':('亥','酉')
}

# 天马 by 年支四组
TIANMA_BY_ZHI_GROUP = {
    '寅午戌':'申', '申子辰':'寅', '巳酉丑':'亥', '亥卯未':'巳'
}

# 火铃 起点（按年支四组），然后以“子时”为起点到生时
HUOLING_GROUP = {
    '申子辰':('寅','戌'), '寅午戌':('丑','卯'), '巳酉丑':('卯','戌'), '亥卯未':('酉','戌')
}

# 左辅右弼、文昌文曲：口诀型函数见下面

# ============== 亮度（庙旺得利平不陷） ==============
# 仅示例：主星及常用辅曜（可按需扩充/替换口径）
BRIGHTNESS_TABLE = {
    # 每颗星一张表：地支→亮度
    '紫微':{'子':'平','丑':'得','寅':'得','卯':'平','辰':'得','巳':'利','午':'旺','未':'旺','申':'得','酉':'得','戌':'得','亥':'平'},
    '天机':{'子':'不','丑':'平','寅':'庙','卯':'旺','辰':'得','巳':'得','午':'平','未':'平','申':'利','酉':'陷','戌':'陷','亥':'平'},
    '太阳':{'子':'庙','丑':'旺','寅':'庙','卯':'旺','辰':'陷','巳':'陷','午':'庙','未':'旺','申':'平','酉':'平','戌':'利','亥':'平'},
    '武曲':{'子':'陷','丑':'平','寅':'平','卯':'平','辰':'旺','巳':'旺','午':'得','未':'得','申':'旺','酉':'旺','戌':'利','亥':'陷'},
    '天同':{'子':'旺','丑':'庙','寅':'利','卯':'利','辰':'陷','巳':'陷','午':'平','未':'平','申':'庙','酉':'旺','戌':'平','亥':'平'},
    '廉贞':{'子':'陷','丑':'陷','寅':'利','卯':'利','辰':'平','巳':'旺','午':'庙','未':'旺','申':'平','酉':'平','戌':'陷','亥':'陷'},
    '天府':{'子':'平','丑':'庙','寅':'同','卯':'旺','辰':'平','巳':'平','午':'旺','未':'庙','申':'同','酉':'旺','戌':'平','亥':'平'},
    '太阴':{'子':'陷','丑':'陷','寅':'平','卯':'平','辰':'旺','巳':'旺','午':'得','未':'得','申':'庙','酉':'庙','戌':'平','亥':'平'},
    '贪狼':{'子':'平','丑':'平','寅':'旺','卯':'旺','辰':'利','巳':'利','午':'平','未':'平','申':'陷','酉':'陷','戌':'平','亥':'平'},
    '巨门':{'子':'利','丑':'旺','寅':'陷','卯':'陷','辰':'平','巳':'平','午':'利','未':'旺','申':'平','酉':'平','戌':'旺','亥':'庙'},
    '天相':{'子':'平','丑':'平','寅':'平','卯':'平','辰':'旺','巳':'旺','午':'平','未':'平','申':'庙','酉':'庙','戌':'平','亥':'平'},
    '天梁':{'子':'平','丑':'平','寅':'平','卯':'平','辰':'庙','巳':'庙','午':'平','未':'平','申':'旺','酉':'旺','戌':'利','亥':'利'},
    '七杀':{'子':'陷','丑':'陷','寅':'旺','卯':'旺','辰':'平','巳':'平','午':'庙','未':'旺','申':'平','酉':'平','戌':'庙','亥':'旺'},
    '破军':{'子':'旺','丑':'旺','寅':'平','卯':'平','辰':'利','巳':'利','午':'平','未':'平','申':'陷','酉':'陷','戌':'平','亥':'平'},
    '文昌':{'子':'利','丑':'平','寅':'平','卯':'庙','辰':'庙','巳':'平','午':'利','未':'平','申':'陷','酉':'陷','戌':'平','亥':'平'},
    '文曲':{'子':'陷','丑':'陷','寅':'平','卯':'平','辰':'庙','巳':'庙','午':'平','未':'平','申':'利','酉':'利','戌':'平','亥':'平'},
    '左辅':{'子':'平','丑':'平','寅':'旺','卯':'旺','辰':'庙','巳':'庙','午':'平','未':'平','申':'陷','酉':'陷','戌':'平','亥':'平'},
    '右弼':{'子':'平','丑':'平','寅':'庙','卯':'庙','辰':'旺','巳':'旺','午':'平','未':'平','申':'陷','酉':'陷','戌':'平','亥':'平'},
}

# ============== 命主/身主 ==============
MINGZHU_BY_YEAR_ZHI = {
    '子':'贪狼','丑':'贪狼','寅':'禄存','卯':'文昌','辰':'廉贞','巳':'武曲','午':'破军',
    '未':'武曲','申':'廉贞','酉':'文昌','戌':'禄存','亥':'贪狼'
}
SHENZHU_BY_YEAR_ZHI = {
    '子':'铃星','午':'铃星','丑':'天相','未':'天相','寅':'天梁','申':'天梁','卯':'天同','酉':'天同','巳':'天机','亥':'天机','辰':'文昌','戌':'文昌'
}

# ==========================
# 宫位与干：定寅首、十二宫干
# ==========================

def palace_gans_by_year_gan(year_gan: str) -> List[str]:
    # 寅宫起干
    gan_at_yin = FIVE_TIGER[year_gan]
    # 从寅开始顺时针排干
    gans = [None]*12
    start = ZHI_IDX['寅']
    for i in range(12):
        gans[(start + i) % 12] = GAN[(GAN_IDX[gan_at_yin] + i) % 10]
    return gans

# ==========================
# 命宫/身宫
# ==========================

def palace_index_of_ming(lunar_month:int, hour_branch:str) -> int:
    # 寅起正月顺至生月 → 基点宫
    base = ZHI_IDX['寅']
    p = (base + (lunar_month-1)) % 12
    # 再自“人生月宫”起子时逆至生时 → 命宫
    h = ZHI_IDX[hour_branch]
    # 子=0 表示逆 0 步；故逆数 h 步
    p = (p - h) % 12
    return p

def palace_index_of_shen(lunar_month:int, hour_branch:str) -> int:
    base = ZHI_IDX['寅']
    p = (base + (lunar_month-1)) % 12
    h = ZHI_IDX[hour_branch]
    p = (p + h) % 12
    return p

# ==========================
# 五行局数（按年干支）
# ==========================

def wuxing_ju(year_gan: str, ming_zhi: str) -> str:
    x = GAN_TO_NUM[year_gan] + ZHI_TO_NUM[ming_zhi]
    while x>5:
        x -= 5
    return WUXING_NAME[x]

# ==========================
# 紫微起星（按局数与“农历日”）
# ==========================

def ziwei_anchor_by_day_and_ju(lunar_day:int, ju:int) -> int:
    # 以寅为 0 基（PALACE ZHI 索引），顺数
    # 规则：商 = d // ju, 余 = d % ju
    # 若余==0：寅顺数 商 到宫
    # 若余为偶：寅顺数 (商+余)
    # 若余为奇：先顺数 商 到 X，再逆数 余 到宫
    start = ZHI_IDX['寅']
    q, r = divmod(lunar_day, ju)
    if r==0:
        steps = q
        idx = (start + (steps-1)) % 12 if steps>=1 else start
        return idx
    if r%2==0:
        steps = q + r
        return (start + (steps-1)) % 12
    else:
        # 先到商，再逆 r
        first = (start + (q-1)) % 12 if q>=1 else start
        return (first - r) % 12

ZIWEI_SERIES = [
    ('紫微',0),('天机',-1),('太阳',+2),('武曲',+3),('天同',+4),('廉贞',+7)
]

# 天府：由紫微落宫→对照映射（寅申同、卯↔丑、辰↔子、巳↔亥、午↔戌、未↔酉）
ZIWEI_TO_TIANFU = {
    '子':'辰','丑':'卯','寅':'寅','卯':'丑','辰':'子','巳':'亥',
    '午':'戌','未':'酉','申':'申','酉':'未','戌':'午','亥':'巳'
}

TIANFU_SERIES = [
    ('天府',0),('太阴',+1),('贪狼',+2),('巨门',+3),('天相',+4),('天梁',+5),('七杀',+6),('破军',+10)
]

# ==========================
# 口诀型杂曜安置
# ==========================

def place_left_right_by_month(lunar_month:int) -> Tuple[str,str]:
    # 辰上起正（正月=1）顺到生月：左辅
    # 戌上起正逆到生月：右弼
    left = (ZHI_IDX['辰'] + (lunar_month-1)) % 12
    right = (ZHI_IDX['戌'] - (lunar_month-1)) % 12
    return ZHI[left], ZHI[right]

def place_wenchang_wenqu_by_hour(hour_branch:str) -> Tuple[str,str]:
    # 戌上起子逆到时：文昌；辰上起子顺到时：文曲
    dz = ZHI_IDX[hour_branch]
    chang = (ZHI_IDX['戌'] - dz) % 12
    qu    = (ZHI_IDX['辰'] + dz) % 12
    return ZHI[chang], ZHI[qu]

def daxian_ranges(gender:str, year_gan:str, start_palace:int, ju:int) -> List[Tuple[int,Tuple[int,int]]]:
    # 返回列表 [(宫索引, (起,止)), ...] 共 12 宫
    # 起始虚岁 = 局数（2/3/4/5/6 → 2/3/4/5/6 岁起）
    start_age = {"水二局":2,"木三局":3,"金四局":4,"土五局":5,"火六局":6}[WUXING_NAME[ju]]
    forward = ((gender=='男' and year_gan in GAN_YANG) or (gender=='女' and year_gan in GAN_YIN))
    seq = [start_palace]
    for i in range(1,12):
        seq.append((seq[-1] + (1 if forward else -1)) % 12)
    age0 = start_age
    return [(p, (age0 + i*10, age0 + i*10 + 9)) for i, p in enumerate(seq)]

def liunian_for_palace(birth_year_zhi:str, palace_zhi:str, up_to:int=60) -> List[int]:
    # 以出生年支逐年递进，求 1..up_to 虚岁时当年支落在 palace_zhi
    # 为匹配常见口径，这里使用 (age-1) 的位移（使本命年为 age=12、24、...）
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
    lunar_month: Optional[int] = None
    lunar_day: Optional[int] = None


def build_chart(inp: InputData) -> Chart:
    # 1) 真太阳时
    ts = true_solar_time(inp.dt_clock, inp.longitude, inp.tz)
    hour_branch = branch_by_true_time(ts)

    # 2) 节气四柱
    y_gz_qi = ganzhi_year_by_jieqi(inp.dt_clock)
    m_gz_qi = month_ganzhi_by_jieqi(inp.dt_clock, y_gz_qi[0])
    d_gz = ganzhi_day(inp.dt_clock)
    h_gz = ganzhi_hour(d_gz[0], hour_branch)

    # 3) 非节气四柱（强制使用 lunardate）
    lunar = LunarInfo(None, None, None, None, None)
    lunar_obj = LunarDate.fromSolarDate(inp.dt_clock.year, inp.dt_clock.month, inp.dt_clock.day)
    lunar.month = lunar_obj.month
    lunar.day = lunar_obj.day
    lunar.is_leap = bool(lunar_obj.isLeapMonth)
    lunar.year_gz = ganzhi_year_by_lunar_year(lunar_obj.year)
    lunar.text = f"{lunar.year_gz}年{('闰' if lunar.is_leap else '')}{LUNAR_MONTH_NAME[lunar.month-1]}月{LUNAR_DAY_NAME[lunar.day-1]}{hour_branch}时"

    # 供“安星/四化/禄存/大限方向/流年”等术数规则使用的年干支 = 农历年干支
    y_gz_lunar = lunar.year_gz
    year_gan_rules = y_gz_lunar[0]
    year_zhi_rules = y_gz_lunar[1]


    # 4) 命宫/身宫
    lm = lunar.month or 1
    ld_day = lunar.day or 1


    ming_idx = palace_index_of_ming(lm, hour_branch)
    shen_idx = palace_index_of_shen(lm, hour_branch)
    shen_zhi = ZHI[shen_idx]

    # 5) 宫干与十二宫名
    gans = palace_gans_by_year_gan(year_gan_rules)
    palaces: List[Palace] = [
        Palace(
            name=PALACE_ORDER[(i - ming_idx) % 12],
            gan=gans[i],
            zhi=ZHI[i],
            tags=['身宫'] if i == shen_idx else []
        )
        for i in range(12)
    ]

    # 6) 五行局
    ju_name = wuxing_ju(year_gan_rules, ZHI[ming_idx])
    ju_num = ju_number(ju_name)

    # 7) 十四主星安置
    # 紫微锚定
    ziwei_pos = ziwei_anchor_by_day_and_ju(ld_day, ju_num)
    # 填紫微系
    for star, off in ZIWEI_SERIES:
        pos = (ziwei_pos + off) % 12
        palaces[pos].placement.main.append(star)
    # 天府锚定
    tf_zhi = ZIWEI_TO_TIANFU[ZHI[ziwei_pos]]
    tianfu_pos = ZHI_IDX[tf_zhi]
    for star, off in TIANFU_SERIES:
        pos = (tianfu_pos + off) % 12
        palaces[pos].placement.main.append(star)

    # 8) 四化（生年）标注
    sihua = SIHUA[year_gan_rules]
    for k, star in sihua.items():  # k in 禄/权/科/忌
        # 找到该星所在宫，加注
        for p in palaces:
            if star in p.placement.main or star in p.placement.assist:
                p.placement.transforms.setdefault(star,[]).append(f"生年{k}")

    # 9) 辅曜
    # 9.1 禄存、羊陀
    lucun_zhi = LUCUN_BY_GAN[year_gan_rules]
    lucun_pos = ZHI_IDX[lucun_zhi]
    palaces[lucun_pos].placement.assist.append('禄存')
    # 羊陀：禄前羊刃当（逆一位两位存在流派差，这里取常见：羊在禄前一、再前一？简化取“前一”），禄后陀罗府（顺一位）
    yang_pos = (lucun_pos - 1) % 12
    tuo_pos  = (lucun_pos + 1) % 12
    palaces[yang_pos].placement.assist.append('擎羊')
    palaces[tuo_pos ].placement.assist.append('陀罗')

    # 9.2 左辅/右弼（按月）
    zf_zhi, yb_zhi = place_left_right_by_month(lm)
    palaces[ZHI_IDX[zf_zhi]].placement.assist.append('左辅')
    palaces[ZHI_IDX[yb_zhi]].placement.assist.append('右弼')

    # 9.3 文昌/文曲（按时）
    wc_zhi, wq_zhi = place_wenchang_wenqu_by_hour(hour_branch)
    palaces[ZHI_IDX[wc_zhi]].placement.assist.append('文昌')
    palaces[ZHI_IDX[wq_zhi]].placement.assist.append('文曲')

    # 9.4 魁钺
    k, y_ = KUIYUE_BY_GAN[year_gan_rules]
    palaces[ZHI_IDX[k]].placement.assist.append('天魁')
    palaces[ZHI_IDX[y_]].placement.assist.append('天钺')

    # 9.5 天马（年支组）
    tianma_zhi = next((z for grp, z in TIANMA_BY_ZHI_GROUP.items() if year_zhi_rules in grp), None)
    if tianma_zhi:
        palaces[ZHI_IDX[tianma_zhi]].placement.assist.append('天马')

    # 9.6 火铃（年支组 + 时）
    for grp, (huo_start, ling_start) in HUOLING_GROUP.items():
        if year_zhi_rules in grp:
            dz = ZHI_IDX[hour_branch]
            huo_pos = (ZHI_IDX[huo_start] + dz) % 12
            ling_pos = (ZHI_IDX[ling_start] + dz) % 12
            palaces[huo_pos].placement.assist.append('火星')
            palaces[ling_pos].placement.assist.append('铃星')
            break

    # 10) 亮度（庙旺…）
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
    ming_zhu = MINGZHU_BY_YEAR_ZHI[year_zhi_rules]
    shen_zhu = SHENZHU_BY_YEAR_ZHI[year_zhi_rules]

    basic = BasicInfo(
        gender=inp.gender,
        longitude=inp.longitude,
        clock_time=inp.dt_clock,
        true_solar_time=ts,
        lunar_text=lunar.text,
        pillars_qi=(y_gz_qi, m_gz_qi, d_gz, h_gz),
        pillars_non_qi=None,
        wuxing_ju=ju_name,
        shen_zhu=shen_zhu,
        ming_zhu=ming_zhu,
        shen_gong_zhi=shen_zhi
    )

    if HAVE_LUNARDATE:
        # 年柱：以“农历年”为界
        ygz = y_gz_lunar
        yg = ygz[0]
        # 月柱（非节气）：寅为正月
        yz_branch = ZHI[(ZHI_IDX['寅'] + (lm - 1)) % 12]
        gan0 = FIVE_TIGER[yg]
        mg = GAN[(GAN_IDX[gan0] + (lm - 1)) % 10]
        # 日柱与时柱：沿用前面通用算法（ganzhi_day / ganzhi_hour）
        basic.pillars_non_qi = (ygz, mg + yz_branch, d_gz, h_gz)


    return Chart(basic=basic, palaces=palaces)

# ==========================
# 文本输出（接近示例样式）
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

    # 以命宫开始逆序输出 12 宫
    # 找到命宫 idx（name==命宫）
    ming_idx = next(i for i, p in enumerate(chart.palaces) if p.name == '命宫')
    order = [(ming_idx + i) % 12 for i in range(12)]
    for idx in order:
        p = chart.palaces[idx]
        head = f"│ ├{p.name}[{p.gan}{p.zhi}]"
        if p.tags:
            head += "["+"、".join(p.tags)+"]"
        lines.append(head)
        # 主星/辅星
        lines.append(format_star_line("主星", p.placement.main, p.placement, include_transforms=True))
        lines.append(format_star_line("辅星", p.placement.assist, p.placement))
        # 大限/流年
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
