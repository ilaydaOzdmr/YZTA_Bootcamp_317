"""
ResilienceOS - CrewAI Ajan Tanimlari
CFO, Tahsilat, Tedarik ve Risk Denetcisi ajanlari.

Kota sorunundan kacmak icin:
- Gemini Flash kullanilir (Pro'dan cok daha az token tuketir)
- max_rpm=10 ile dakika basi istek siniri uygulanir
"""
import os
from crewai import Agent, LLM
from .tools import (
    nakit_raporu_tool,
    riskli_fatura_tool,
    stok_riski_tool,
    aksiyon_onerisi_tool,
    genel_durum_tool,
)

_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY", ""),
    max_rpm=10,
)


cfo_ajan = Agent(
    role="CFO - Finans Direktoru",
    goal=(
        "Nakit akisi projeksiyonunu incele. Kriz riski varsa tarihini ve "
        "buyuklugunu net sekilde raporla. Gercek sayilara dayan, yorum katma."
    ),
    backstory=(
        "KOBİ finansmani uzmanı. Nakit döngüsünü, alacak/borc dengesini "
        "ve likidite riskini sayisal olarak degerlendirir."
    ),
    tools=[nakit_raporu_tool, genel_durum_tool],
    llm=_llm,
    verbose=False,
    allow_delegation=False,
)

tahsilat_ajan = Agent(
    role="Tahsilat Uzmani",
    goal=(
        "En riskli acik faturalari belirle. Her fatura icin beklenen gecikme "
        "ve onerilecek erken odeme tesviki miktarini hesapla."
    ),
    backstory=(
        "Alacak yonetimi ve tahsilat kampanyalari konusunda deneyimli uzman. "
        "Musteri segmentine ve gecmis odeme davranisinа gore oncelik belirler."
    ),
    tools=[riskli_fatura_tool],
    llm=_llm,
    verbose=False,
    allow_delegation=False,
)

tedarik_ajan = Agent(
    role="Tedarik Zinciri Uzmani",
    goal=(
        "Kritik stok tukenmelerini tespit et. Hangi urunun ne zaman bitecegini "
        "ve kesintinin maliyet etkisini raporla."
    ),
    backstory=(
        "Uretim planlama ve tedarik zinciri yonetimi uzmanı. "
        "Tükenme tarihlerini ve yeniden siparis noktalarini talep tahminiyle hesaplar."
    ),
    tools=[stok_riski_tool],
    llm=_llm,
    verbose=False,
    allow_delegation=False,
)

risk_denetcisi = Agent(
    role="Risk Denetcisi ve Orkestrator",
    goal=(
        "CFO, Tahsilat ve Tedarik ajanlarindan gelen bulgulari birlestir. "
        "Hangi aksiyonun krizi kac puanlik olasilikla cozduğunu goster. "
        "Tek bir net eylem plani sun."
    ),
    backstory=(
        "Ust yonetim danismani. Finansal, operasyonel ve tedarik risklerini "
        "butunlesik degerlendirir; sayisal temelli aksiyon onceliklendirir."
    ),
    tools=[aksiyon_onerisi_tool],
    llm=_llm,
    verbose=False,
    allow_delegation=False,
)
