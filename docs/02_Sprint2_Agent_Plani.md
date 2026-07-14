# Sprint 2 — Multi-Agent Katman Planı (taslak)

> Amaç: Sprint 1'de kurduğumuz modelleri + şok simülasyonunu, birbiriyle konuşan yapay
> zeka ajanlarına bağlamak. Ajanlar `src/core/twin_api.py`'deki hazır fonksiyonları çağırır
> (framework-bağımsız — CrewAI, LangChain veya düz Python ile çalışır).

## Ajan mimarisi

| Ajan | Görevi | Çağırdığı API fonksiyonu |
|------|--------|--------------------------|
| **CFO Agent** | Nakit darboğazını tespit eder | `cash_forecast()`, `get_overview()` |
| **Tahsilat Agent** | Riskli faturaları sıralar, tahsilat stratejisi + mail taslağı | `invoice_risk()` |
| **Tedarik Zinciri Agent** | Stok tükenme / tedarikçi gecikme riskini analiz eder | `stock_risk()` |
| **Risk Denetçisi Agent** | Şok senaryolarını ve önerilerin uygunluğunu denetler | `simulate()` |
| **Orkestratör** | Ajan sırasını yönetir, çelişkileri çözer, aksiyon planı hazırlar | `recommend()`, `situation_report()` |

## Demo akışı (Sprint 1'de üretilen gerçek çıktılarla)

```
"Bugün" = 31 Mayıs 2025 (kriz henüz olmadı)

1. Orkestratör -> situation_report()
2. CFO         -> cash_forecast(): "18 Haziran'da kasa -46.701 TL, KRİZ"
                  (gerçekleşen: -59.780 TL @ 18 Haziran -> tarih sapması 0 gün)
3. Tahsilat    -> invoice_risk(): ABC'nin 180.000 TL faturası ~15 gün gecikecek
4. Tedarik     -> stock_risk(): kritik Ham Kumaş 30 Haziran'da tükeniyor (KRİTİK)
5. Risk Den.   -> simulate("3_kur_soku"): kur %10 artarsa kriz -313.901 TL'ye derinleşir
6. Orkestratör -> recommend(): KOMBİNE ÇÖZÜM (erken ödeme + tedarikçi bölme)
                  -> kasa +374.621 TL, kriz kapanır
```

**Not:** İyimser senaryoda (tüm tahsilatlar vadesinde) kasa +229.067 TL, yani kriz yok.
Krizin sebebi **tahsilat gecikmesi** — bu, Tahsilat Agent'ın aksiyonunu doğrudan gerekçelendirir.

## Modellerin çıktıya etkisi (jüri sorabilir)

| Model | Çıktıya etkisi |
|-------|----------------|
| **Fatura gecikme (LightGBM)** | Nakit projeksiyonundaki **tahsilat tarihlerini** belirler → krizi o sürükler |
| **Talep tahmini (LightGBM)** | Stok tükenme tarihini **özyinelemeli tahminle** üretir |
| Nakit projeksiyonu | Deterministik "bilinen defter" (direct method); **ML katkısı tahsilat zamanlaması** |

Bu ayrımı dürüstçe anlatıyoruz: krizi bulan şey ML tahsilat zamanlaması + muhasebe defteri.
"ML krizi keşfetti" gibi bir iddia çürütülebilir; bu çerçeve savunulabilir.

## Teknik kararlar (toplantıda netleştirilecek)

- **Framework:** Öneri **CrewAI** (rol/görev ayrımı net; `twin_api` fonksiyonları doğrudan
  `@tool` olarak sarılır). Alternatif: LangGraph.
- **LLM:** API anahtarı kimde olacak?
- **RAG / ortak hafıza:** ChromaDB; CrewAI'nin yerleşik memory'si başlangıç için yeterli
  (sürüm sabitlenmeli — güncel sürümlerde varsayılan backend değişti).
- **Mail taslağı:** Tahsilat ajanı, şirket tonuna uygun hatırlatma üretir.

## Hazır altyapı (Sprint 1)

- `src/core/twin_api.py` — 6 temiz fonksiyon (JSON döner, test edildi)
- `src/simulation/shock_simulation.py` — aksiyon motoru
- 2 eğitilmiş model + doğrulanmış (gerçek Kaggle dağılımlarına kalibre) sentetik veri

Sprint 2'de sıfırdan değil, hazır API üzerine **ajan sarma** işi kalıyor.
