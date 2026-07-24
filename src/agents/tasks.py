"""
ResilienceOS - CrewAI Gorev Tanimlari
Her ajan icin sinirli, belirli bir ciktiya sahip gorevler.
Acik uclu gorevler LLM'i gereksiz dongulere sokar; bu dosya bunu onler.
"""
from crewai import Task
from .agents import cfo_ajan, tahsilat_ajan, tedarik_ajan, risk_denetcisi


nakit_gorevi = Task(
    description=(
        "Nakit akisi aracini kullan. Kriz tarihini, beklenen en dusuk "
        "kasa bakiyesini ve kriz olasiligini raporla. "
        "Ek yorum veya oneri ekleme, sadece bulgulari aktar."
    ),
    expected_output=(
        "Kriz tarihi, en dusuk bakiye (TL), kriz olasiligi (%), "
        "tampon ihlali olasiligi (%) degerlerini iceren kisaca ozet."
    ),
    agent=cfo_ajan,
)

tahsilat_gorevi = Task(
    description=(
        "Riskli fatura aracini kullan. En fazla 5 kritik faturanin "
        "musteri ID'sini, tutarini ve tahmini gecikme gununu listele. "
        "Tahsilat kampanyasi icin hangi musterilerin onceliklendirilecegini belirt."
    ),
    expected_output=(
        "En riskli 5 fatura: musteri, tutar, gecikme gun tahmini ve "
        "onerilecek erken odeme tesviki orani (%) iceren liste."
    ),
    agent=tahsilat_ajan,
    context=[nakit_gorevi],
)

tedarik_gorevi = Task(
    description=(
        "Stok tukenme aracini kullan. KRITIK ve YUKSEK riskli urunleri listele. "
        "Her urun icin tukenme tarihi ve 30 gunluk kesinti maliyetini ver."
    ),
    expected_output=(
        "KRITIK/YUKSEK riskli urunler: urun adi, tukenme tarihi, "
        "30g kesinti maliyeti (TL) iceren liste."
    ),
    agent=tedarik_ajan,
    context=[nakit_gorevi],
)

orkestrasyon_gorevi = Task(
    description=(
        "Aksiyon onerisi aracini kullan. CFO, Tahsilat ve Tedarik "
        "bulgularini dikkate alarak: (1) mevcut durum ozeti, "
        "(2) en etkili 3 aksiyon ve kriz olasiligi degisimi, "
        "(3) oncelikli tek eylem plani sun."
    ),
    expected_output=(
        "Yonetici ozeti: kriz riski, onerilen 3 aksiyon (her biri icin "
        "onceki/sonraki kriz olasiligi), ve 1 haftalik eylem takvimi."
    ),
    agent=risk_denetcisi,
    context=[nakit_gorevi, tahsilat_gorevi, tedarik_gorevi],
)
