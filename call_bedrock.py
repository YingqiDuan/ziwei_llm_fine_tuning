import boto3
import json
import time
from botocore.exceptions import ClientError
from prompts import DEFAULT_SYSTEM_PROMPT

region = "us-east-1"
model_arn = ""

bedrock = boto3.client("bedrock-runtime", region_name=region)

body = {
    "messages": [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "紫微斗数命盘\n│\n├基本信息\n│ │\n│ ├性别 : 女\n│ ├阳历 : 1978-11-11 卯时 (05:00~07:00)\n│ ├农历 : 一九七八年十月十一\n│ ├干支 : 戊午 癸亥 丁丑 癸卯\n│ ├命主 : 廉贞\n│ ├身主 : 火星\n│ ├命宫地支 : 申\n│ ├身宫地支 : 寅\n│ └五行局 : 木三局\n│\n├命盘十二宫\n│ │\n│ ├命宫[庚申]\n│ │ ├主星 : 廉贞[庙]\n│ │ ├辅星 : 天马,地空\n│ │ ├大限 : 3~12虚岁 (庚申)\n│ │ └流年 : 9,21,33,45,57…\n│ │\n│ ├父母[辛酉]\n│ │ ├主星 : 无\n│ │ ├辅星 : 无\n│ │ ├大限 : 113~122虚岁 (辛酉)\n│ │ └流年 : 8,20,32,44,56…\n│ │\n│ ├福德[壬戌]\n│ │ ├主星 : 破军[旺]\n│ │ ├辅星 : 无\n│ │ ├大限 : 103~112虚岁 (壬戌)\n│ │ └流年 : 7,19,31,43,55…\n│ │\n│ ├田宅[癸亥]\n│ │ ├主星 : 天同[庙]\n│ │ ├辅星 : 无\n│ │ ├大限 : 93~102虚岁 (癸亥)\n│ │ └流年 : 6,18,30,42,54…\n│ │\n│ ├官禄[甲子]\n│ │ ├主星 : 武曲[旺],天府[庙]\n│ │ ├辅星 : 无\n│ │ ├大限 : 83~92虚岁 (甲子)\n│ │ └流年 : 5,17,29,41,53…\n│ │\n│ ├仆役[乙丑]\n│ │ ├主星 : 太阳[不],太阴[庙][权]\n│ │ ├辅星 : 左辅,右弼[科],天魁\n│ │ ├大限 : 73~82虚岁 (乙丑)\n│ │ └流年 : 4,16,28,40,52…\n│ │\n│ ├迁移[甲寅][身宫]\n│ │ ├主星 : 贪狼[平][禄]\n│ │ ├辅星 : 地劫\n│ │ ├大限 : 63~72虚岁 (甲寅)\n│ │ └流年 : 3,15,27,39,51…\n│ │\n│ ├疾厄[乙卯]\n│ │ ├主星 : 天机[旺][忌],巨门[庙]\n│ │ ├辅星 : 无\n│ │ ├大限 : 53~62虚岁 (乙卯)\n│ │ └流年 : 2,14,26,38,50…\n│ │\n│ ├财帛[丙辰]\n│ │ ├主星 : 紫微[得],天相[得]\n│ │ ├辅星 : 火星[陷],陀罗[庙]\n│ │ ├大限 : 43~52虚岁 (丙辰)\n│ │ └流年 : 1,13,25,37,49…\n│ │\n│ ├子女[丁巳]\n│ │ ├主星 : 天梁[陷]\n│ │ ├辅星 : 禄存\n│ │ ├大限 : 33~42虚岁 (丁巳)\n│ │ └流年 : 12,24,36,48,60…\n│ │\n│ ├夫妻[戊午]\n│ │ ├主星 : 七杀[旺]\n│ │ ├辅星 : 铃星[庙],擎羊[陷]\n│ │ ├大限 : 23~32虚岁 (戊午)\n│ │ └流年 : 11,23,35,47,59…\n│ │\n│ ├兄弟[己未]\n│ │ ├主星 : 无\n│ │ ├辅星 : 文昌[利],文曲[旺],天钺\n│ │ ├大限 : 13~22虚岁 (己未)\n│ │ └流年 : 10,22,34,46,58…\n│ │\n│\n└"
        }
    ],
    "temperature": 0,
    "top_p": 1.0
}

max_retries = 10
backoff = 5

def extract_text(result: dict) -> str:
    if "choices" not in result or not result["choices"]:
        return ""

    choice = result["choices"][0]
    message = choice.get("message", {})

    content = message.get("content")
    reasoning = message.get("reasoning_content")
    refusal = message.get("refusal") or choice.get("refusal")

    # 1）优先用 content（如果是字符串）
    if isinstance(content, str) and content.strip():
        return content.strip()

    # 2）否则用 reasoning_content（你的模型现在就是走这里）
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # 3）如果有拒绝信息
    if isinstance(refusal, str) and refusal.strip():
        return f"[model refusal] {refusal.strip()}"

    # 4）兜底：直接把整个结构吐出来，方便调试
    return json.dumps(result, ensure_ascii=False, indent=2)


for attempt in range(1, max_retries + 1):
    try:
        response = bedrock.invoke_model(
            modelId=model_arn,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        text = extract_text(result)
        print(text)
        print("usage:", result.get("usage"))
        break

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ModelNotReadyException":
            print(f"[attempt {attempt}] Model not ready yet, sleeping {backoff}s ...")
            time.sleep(backoff)
            backoff *= 2
            continue
        else:
            raise
else:
    raise RuntimeError("Model still not ready after retries")