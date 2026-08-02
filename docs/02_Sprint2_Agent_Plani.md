# Multi-Agent Katman Planı (Sprint 2 tasarımı)

> **Durum:** Bu plan Sprint 3'te uygulandı — `src/agents/` altında CrewAI + Gemini 2.0 Flash
> ile 4 ajan yazıldı. Uygulanan katmanın ayrıntısı: [`04_Multi_Agent_Katmani.md`](04_Multi_Agent_Katmani.md).
>
> Amaç: Sprint 1'de kurduğumuz modelleri + şok simülasyonunu, birbiriyle konuşan yapay
> zeka ajanlarına bağlamak. Ajanlar `src/core/twin_api.py`'deki hazır fonksiyonları çağırır.

## Ajan mimarisi

| Ajan | Görevi | Çağırdığı API fonksiyonu |
|------|--------|--------------------------|
| **CFO Agent** | Nakit darboğazını tespit eder | `cash_forecast()`, `get_overview()` |
| **Tahsilat Agent** | Riskli faturaları sıralar, tahsilat stratejisi + mail taslağı | `invoice_risk()` |
| **Tedarik Zinciri Agent** | Stok tükenme / tedarikçi gecikme riskini analiz eder | `stock_risk()` |
| **Risk Denetçisi Agent** | Şok senaryolarını ve önerilerin uygunluğunu denetler | `simulate()` |
| **Orkestratör** | Ajan sırasını yönetir, çelişkileri çözer, aksiyon planı hazırlar | `recommend()`, `situation_report()` |

## Demo akışı (Sprint 1'de üretilen gerçek çıktılarla; ufuk 30 gün)

```
"Bugün" = 31 Mayıs 2025 (kriz henüz olmadı)

1. Orkestratör -> situation_report()
2. CFO         -> cash_forecast(): "eksiye düşme olasılığı %56, en olası tarih 2025-06-18"
                  (gerçekleşen: -59.780 TL @ 18 Haziran -> tarih sapması 0 gün)
3. Tahsilat    -> invoice_risk(): ABC'nin 180.000 TL faturası gecikecek (en riskli)
4. Tedarik     -> stock_risk(): kritik Ham Kumaş 30 Haziran'da tükeniyor (KRİTİK)
5. Risk Den.   -> simulate("kur_soku"): kur şoku uygulanırsa kriz olasılığı %100
6. Orkestratör -> recommend(): KOMBİNE ÇÖZÜM (erken ödeme + tedarikçi bölme)
                  -> kriz olasılığı %56 -> %0
```

**Not:** İyimser senaryoda (tüm tahsilatlar vadesinde) kasa +299.204 TL. kriz yok.
Kapsam: projeksiyon 30 günlük ufukta ve yalnızca **bilinen defter** (cutoff'a kadar kesilmiş
faturalar) üzerinden yapılır; "şirket sonsuza dek güvenli" gibi bir iddia YOKTUR.
Krizin sebebi **tahsilat gecikmesi** — bu, Tahsilat Agent'ın aksiyonunu doğrudan gerekçelendirir.

## Modellerin çıktıya etkisi

| Model | Çıktıya etkisi |
|-------|----------------|
| **Fatura gecikme (LightGBM)** | Nakit projeksiyonundaki **tahsilat tarihlerini** belirler → krizi o sürükler |
| **Talep tahmini (LightGBM)** | Stok tükenme tarihini **özyinelemeli tahminle** üretir |
| Monte Carlo | Modelin hata dağılımından örnekleyerek **kriz olasılığını** hesaplar |
| Nakit projeksiyonu | Deterministik "bilinen defter" (direct method); **ML katkısı tahsilat zamanlaması** |

Krizi belirleyen şey, ML'in tahsilat zamanlaması ile muhasebe defterinin birleşimidir;
ML tek başına "krizi keşfetmez". Bu ayrım dokümantasyonda bilinçli olarak korunur.

## Teknik kararlar (uygulandı)

- **Framework:** ✅ **CrewAI** seçildi — `twin_api` fonksiyonları `@tool` olarak sarıldı.
- **LLM:** ✅ **Gemini 2.0 Flash** (`.env` içinde `GEMINI_API_KEY`).
- **RAG / ortak hafıza:** başlangıç için kullanılmadı; orkestratör diğer ajanların çıktısını
  `context` olarak alıyor. (RAG/ChromaDB backlog'da opsiyonel.)
- **Mail taslağı:** Tahsilat ajanı için planlandı — henüz uygulanmadı.

## Hazır altyapı (Sprint 1)

- `src/core/twin_api.py` — 6 temiz fonksiyon (JSON döner, test edildi)
- `src/simulation/shock_simulation.py` — aksiyon motoru
- 2 eğitilmiş model + doğrulanmış (gerçek Kaggle dağılımlarına kalibre) sentetik veri

Sprint 2'de sıfırdan değil, hazır API üzerine **ajan sarma** işi kalıyor.
