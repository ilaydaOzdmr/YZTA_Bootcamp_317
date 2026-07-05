# ResilienceOS — Dataset Araştırması & Veri İş Akışı (v2, Doğrulanmış)

> Sprint 1 hazırlık dokümanı · Güncelleme: 30 Haziran 2026
> **v2 notu:** 7 açıdan çoklu-ajan derin araştırma + her veri setinin bağımsız erişilebilirlik/şema doğrulaması (40 ajan, 264 web araması) sonucu. Doğrulama düzeltmeleri ⚠️ ile işaretlendi.

---

## 0. Strateji — Karar (değişmedi, güçlendi)

Proje bir **Dijital İkiz**: tek kurgusal KOBİ'nin faturaları + banka + stok + tedarikçi verisi **ilişkisel olarak tutarlı** olmalı. Hiçbir açık veri seti bu dördünü tek şirket için birlikte sunmuyor.

**Hibrit strateji:**
1. **Gerçek veri setlerinden dağılım öğren** (gecikme, talep, mevsimsellik, tedarik gecikme).
2. **Faker + SDV ile tek kurgusal şirketin ilişkisel verisini üret** (SQLite), gerçek dağılıma kalibre et.
3. **Gerçek makro seriyi (TCMB EVDS) bindir** — şok simülasyonu için.

Araştırma bu stratejiyi doğruladı: en güçlü akademik referans **arXiv 2511.03631 (SME Financial Management System)** birebir bu yolu izliyor — IBM + HighRadius Kaggle setlerini kullanıp, soğuk başlangıçta (cold-start, az veri) modüler yaklaşımı klasik modellere üstün buluyor (**MAPE %11.85** vs Prophet %159, SVR %166). Yani "az veri + sentetik + modüler" yaklaşımı KOBİ gerçekliğine en uygunu — jüriye savunulabilir.

---

## 1. ÖNCELİKLİ VERİ SETLERİ (modül bazlı en iyi seçimler)

### 1.1 ⭐ Fatura / Tahsilat Riski

| Veri | Erişim | Boyut | Hedef | Durum |
|------|--------|-------|-------|-------|
| **Payment Date Prediction for Invoices** (`pradumn203`) | Kaggle | ~50.000 (train+açık faturalar) | gün gecikme / ödeme tarihi | ✅ **confirmed** — birincil öneri |
| **Payment Date Dataset HighRadius** (`rajattomar132`) | Kaggle | ~50.000 | `clear_date - due_in_date` | ⚠️ partial |
| **IBM Finance Factoring** (`hhenry`) | Kaggle | ~2.466 | `DaysLate` | ✅ confirmed — temiz baseline |

**HighRadius şeması:** `business_code, cust_number, name_customer, clear_date, due_in_date, total_open_amount, cust_payment_terms, baseline_create_date, invoice_currency, is_open_invoice, document_type, posting_id`. Hedef = `clear_date − due_in_date` (regresyon) veya gec/zamanında (sınıflandırma).

> ⚠️ **Doğrulama düzeltmeleri:** (1) HighRadius verisi gerçekte **büyük kurumsal B2B** (SAP/HighRadius müşterileri), "KOBİ verisi" değil — model şablonu olarak kullan ama jüriye "KOBİ verisi" deme. (2) Kolon adı `isOpen` değil **`is_open_invoice`**. (3) Lisans Kaggle'da net değil; indirmeden önce "License" alanına bak.

**IBM şeması (kesin):** `countryCode, customerID, PaperlessDate, invoiceNumber, InvoiceDate, DueDate, InvoiceAmount, Disputed, SettledDate, PaperlessBill, DaysToSettle, DaysLate`. → `DaysLate` regresyon. ABC Tekstil senaryonun birebir altyapısı.

**Kalibrasyon ground-truth (gerçek, jüri için altın değerinde):**
- **UK Payment Practices Reporting** (GOV.UK) — ✅ confirmed, **Open Government Licence** (ticari kullanıma açık!), CSV toplu indirme (`/export/csv/`). Firma düzeyinde gerçek "ortalama ödeme günü, %30/60/61+ gün içinde ödenen, geç ödeme %". Sentetik gecikme dağılımını gerçek istatistiğe kalibre etmek için.
- **EU Payment Observatory / Intrum** — sektör/ülke kırılımında DSO, B2B ödeme açığı (kalibrasyon için ikincil).

**Feature engineering şablonu (arXiv 2008.07363 & 1912.10828):** Müşteri bazlı tarihsel agregat feature'lar — `total_paid_invoices, sum_amount_late, total_invoices_late, average_days_late, std_dev_invoices_late, payment_frequency_difference, total_outstanding`. Etiket: vade + 5 gün eşiği. Bu liste sentetik üreticinin de hedef desenidir.

---

### 1.2 ⭐ Stok & Tedarik Riski

| Veri | Erişim | Boyut | Notlar | Durum |
|------|--------|-------|--------|-------|
| **Retail Store Inventory Forecasting** (`anirudhchauhan`) | Kaggle | ~73.100 satır, 15 kolon, günlük | stok+satış+talep+hava+promo birlikte | ✅ confirmed |
| **DataCo Smart Supply Chain** (`shashwatwork`) | Kaggle | 180.519 satır, 53 kolon | **`Late_delivery_risk`** hazır hedef (~%55 geç) | ✅ confirmed — tedarik riski için **en güçlü** |
| **Store Sales – Favorita** (Kaggle comp) | Kaggle | ~3M satır, **gerçek** | petrol fiyatı + tatil → makro-talep kesişimi | ✅ confirmed |
| **Logistics & Supply Chain** (`datasetengineer`) | Kaggle | binler/on binler | **Supplier Reliability Score (0-1)** + lead time | ✅ confirmed |

**Retail Inventory şeması:** `Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality`.

**DataCo (tedarik riski):** `Days for shipping (real)`, `Days for shipment (scheduled)`, `Delivery Status`, `Late_delivery_risk` (ikili hedef), `Shipping Mode`, `Order Profit Per Order`. → "tedarikçi 2 hafta gecikirse" senaryosunun gerçek veriyle desteklenmiş hali.

> ⚠️ **Doğrulama düzeltmesi (v1 hatası):** "Store Item Demand Forecasting" (`dhrubangtalukdar`) **10 mağaza / 913K satır DEĞİL** — bu re-upload aslında **50 mağaza × 50 ürün ≈ 4.5M satır**, ek kolonlar `price, promo, weekday, month` içeriyor. Orijinal 10-mağaza/913K olan, ayrı Kaggle yarışmasıdır. Karıştırma.

**🇹🇷 Türkiye gerçek perakende (yerel senaryo için):**
- **1M rows Turkish Market Sales** (`omercolakoglu`) — ~1M satır, gerçek KOBİ market, Türkçe ürün adları (eşleştirme/temizlik gerekir).
- **Customer Shopping – Istanbul Retail** (`mehmettahiraslan`) — ~99K, gerçek İstanbul AVM verisi.

---

### 1.3 ⭐ Banka Hareketleri & Anomali

| Veri/Araç | Erişim | Boyut | Hedef | Durum |
|-----------|--------|-------|-------|-------|
| **PaySim** (`ealaxi/paysim1`) | Kaggle, CC BY-SA 4.0 | ~6.3M işlem | `isFraud` + bakiye tutarsızlığı | ✅ confirmed |
| **Sparkov Generator** (`namebrandon`) | GitHub, açık | yapılandırılabilir | Faker tabanlı işlem üretici | ✅ confirmed |
| **Sparkov Dataset** (`kartik2112`) | Kaggle, CC0 | ~1.85M | `is_fraud` + merchant/kategori | ✅ confirmed |
| **Credit Card Fraud** (`mlg-ulb`) | Kaggle | 284.807 | `Class` (dengesiz) | ✅ confirmed |
| **Bank Statement** (`apoorvwatsky`) | Kaggle | ~116K | NLP merchant kategorizasyon | finding |

**PaySim şeması:** `step, type(CASH-IN/OUT/DEBIT/PAYMENT/TRANSFER), amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud`. Anomali modülü için `IsolationForest` referansı; bakiye-öncesi/sonrası tutarsızlık deseni senin sentetik banka defterine örnek.

---

### 1.4 ⭐ Türkiye Makro Katmanı — TCMB EVDS (somut, doğrulanmış)

**Python paketi (öneri):** `pip install evds --upgrade` (fatihmete, MIT). Alternatif: `evdspy` (cache'li), `borsapy` (çoğu fonksiyon key'siz), `tcmb-py` (wildcard).

**✅ Doğrulanmış seri kodları:**
| Seri | Kod |
|------|-----|
| USD/TRY (alış) | `TP.DK.USD.A.YTL` |
| USD/TRY (satış) | `TP.DK.USD.S.YTL` |
| EUR/TRY (alış) | `TP.DK.EUR.A.YTL` |
| TÜFE (CPI) | `TP.FG.J0` |
| Ağırlıklı Ort. Fonlama Maliyeti | `TP.APIFON4` |
| TL gecelik referans faiz (TLREF) | `TP.BISTTLREF.KAPANIS` |

**Kullanım:**
```python
from evds import evdsAPI
evds = evdsAPI('API_KEY')
df = evds.get_data(['TP.DK.USD.A.YTL', 'TP.FG.J0'],
                   startdate='01-01-2024', enddate='01-01-2026',
                   frequency=5,        # 1=günlük, 5=aylık, 8=yıllık
                   formulas=3)         # 0=seviye, 1=% değişim, 3=yıllık % değişim
```

> ⚠️ **KRİTİK (05/04/2024 sonrası):** API key artık URL parametresi değil **HTTP header**'da gönderiliyor (`headers={'key': api_key}`). Güncel paket sürümü şart; eski sürümler kırılır. API key: evds3.tcmb.gov.tr → kayıt → Profil → "API Key Kopyala" (ücretsiz, 5 dk).

**Yedek/çapraz doğrulama:** `yfinance` (`USDTRY=X`, key gerekmez) — TCMB resmi kuruna karşı kontrol; `wbgapi` (World Bank, key gerekmez) — `FP.CPI.TOTL.ZG` enflasyon, uluslararası kıyas.

---

## 2. Sentetik Üretim Araçları — Karar

| Araç | Rol | İlişkisel/FK | Lisans | Karar |
|------|-----|--------------|--------|-------|
| **Faker** | Temel üretici (isim, IBAN, tarih, tutar) | Manuel (parent ID listesi + choice) | MIT | ✅ **Birincil** |
| **SDV** (HMASynthesizer) | Çok-tabloli FK bütünlüğü KORUYARAK üretim | ✅ Dahili (one-to-many) | ⚠️ BSL (ticari sınırlı) | 🔶 Gerçek veriden öğrenip çoğaltmak istersen |
| **SDV** (PARSynthesizer) | Sequential/zaman serisi üretim | — | BSL | 🔶 Banka zaman serisi için |
| **Mimesis** | Faker'in ~12x hızlı alternatifi | Kısmi (şema referansı) | MIT | 🔶 Yüz binlerce satır için |
| CTGAN | Tek-tablo GAN, gerçek veriden | Hayır | BSL | ❌ Gerçek kayıt yoksa aşırı |
| Gretel | Zaman serisi GAN | — | ⚠️ Source-available | ❌ **18 Şub 2026'da arşivlendi** |

**Net öneri:** **Faker (temel) + SDV HMASynthesizer (ilişkisel tutarlılık)**. Faker ile kurgusal şirketi sıfırdan kur; ilişkisel bütünlük ve gerçekçilik için SDV'nin metadata + HMA katmanını ekle. Banka defteri için running-balance deseni: [Faker financial transaction örneği](https://python-fiddle.com/examples/faker-financial-transactions) (`balance_after` alanı).

> ⚠️ **Düzeltmeler:** mockdatafaker.com **Python Faker DEĞİL**, tarayıcı-tabanlı JS aracı — Python pipeline'ına entegre olmaz. Mimesis 34 değil **47 lokal** destekler. SDV/CTGAN artık **MIT değil BSL** — bootcamp non-ticari olduğu için sorun değil, ama bil.

---

## 3. Multi-Agent + RAG Hafıza Referansları

| Kaynak | Ne gösteriyor | ResilienceOS karşılığı |
|--------|---------------|------------------------|
| **wyne1/financial-analyst-crewai** | 4 ajan: Data Analyst, Strategy, Advisor, **Risk Advisor** | CFO/Tedarik/Tahsilat/**Risk Denetçisi** rol ayrımına en yakın şablon ✅ |
| **botextractai/ai-crewai-multi-agent** | CrewAI + **FAISS** RAG, sıralı ajanlar | Fatura/banka belgelerini RAG'a gömüp ajana besleme ✅ |
| **CrewAI Memory** | `memory=True` | RAG ortak hafıza ⚠️ (aşağı bak) |
| **MLAT framework** (arXiv 2602.14295) | ML modelini ajana **tool** olarak verme | ML tahmin çıktısını → ajan context'i (prediction-to-context) |
| **Mem0 for CrewAI** | Production'da kalıcı hafıza | Ölçeklenince ChromaDB yerine |

> ⚠️ **CrewAI Memory düzeltmesi:** Klasik mimari Short-Term **ChromaDB** + Long-Term **SQLite** + Entity idi; güncel CrewAI bunu tek `Memory` sınıfı + varsayılan **LanceDB** ile değiştirdi. Sprint 2'de **sürüm pinle** ve hangi backend'i kullandığını netleştir (ChromaDB istiyorsan eski sürüm/explicit config).

---

## 4. Önerilen Veri Pipeline (güncel)

```
[1] KAYNAK
    ├─ Fatura: Payment Date (pradumn203) + IBM Finance Factoring  → gecikme dağılımı
    ├─ Stok:   Retail Inventory + DataCo (tedarik riski) + Favorita
    ├─ Banka:  PaySim (anomali deseni referansı)
    ├─ Makro:  TCMB EVDS (canlı API) → USD/TRY, TÜFE, faiz
    └─ Kalibrasyon: UK Payment Practices (gerçek ödeme davranışı)
                  │
[2] PROFİL & KALİBRASYON
    └─ Gecikme/talep/tedarik-gecikme dağılımlarını istatistiksel çıkar
                  │
[3] SENTETİK ÜRETİM  (Faker + SDV HMA → SQLite digital_twin.db)
    └─ Tek kurgusal KOBİ: customers→invoices→bank_ledger→inventory→suppliers (FK tutarlı)
       gecikme lojiği §1.1 feature/dağılımına kalibre; demo darboğazı kasıtlı göm
                  │
[4] FEATURE ENGINEERING
    ├─ Fatura: average_days_late, sum_amount_late, payment_frequency_difference, disputed
    ├─ Nakit:  lag(t-1,t-7,t-30), rolling(7g mean,30g max,90g trend), dow/month/holiday
    └─ Stok:   sales velocity, days-of-cover, supplier lead-time, Late_delivery_risk
                  │
[5] MODEL (Jupyter — Sprint 1 base)
    ├─ Fatura gecikme: LightGBM/CatBoost (RMSE / F1)
    ├─ Nakit akışı:    Prophet / LightGBM multi-step (MAPE)   ← cold-start: modüler > ARIMA/Prophet (arXiv 2511.03631)
    └─ Stok/talep:     LightGBM / Prophet (RMSE)
                  │
[6] AGENT + RAG (Sprint 2)  → ML çıktısı = ajan tool/context (MLAT deseni); CrewAI + ChromaDB/FAISS
                  │
[7] ŞOK SİMÜLASYONU  → EVDS makro değiştir (kur +%10) → pipeline yeniden koş → counterfactual kasa
```

---

## 5. Sprint 1 Görev Listesi (→ 5 Temmuz) — güncel

- [ ] **T1.** Kaggle'dan indir + EDA: `pradumn203/payment-date-prediction`, `hhenry/finance-factoring`, `anirudhchauhan/retail-inventory`, `shashwatwork/dataco-supply-chain`. Gecikme + talep + tedarik-gecikme dağılımlarını çıkar.
- [ ] **T2.** TCMB EVDS API key al → `evds` ile `TP.DK.USD.A.YTL` + `TP.FG.J0` + `TP.APIFON4` çek → `macro_daily` tablosu. (Header değişikliğine dikkat.)
- [ ] **T3.** UK Payment Practices CSV indir → gerçek ödeme-günü dağılımını kalibrasyon tablosu yap.
- [ ] **T4.** SQLite şema (§ aşağıda) + Faker jeneratör; gecikmeyi T1 dağılımına kalibre et; demo darboğazını göm. (Gerekirse SDV HMA ile ilişkisel sağlamlaştır.)
- [ ] **T5.** Banka defterini fatura/gider tablolarından **türet** (bağımsız üretme).
- [ ] **T6.** Base model: fatura `DaysLate` (LightGBM), notebook'ta eğit, RMSE + feature importance raporla, GitHub'a push.
- [ ] **T7.** README veri sözlüğü + bu workflow özeti + kaynak/lisans listesi.

**SQLite şeması:** `customers`, `suppliers`, `products`, `sales_invoices`, `purchase_invoices`, `bank_ledger` (türetilmiş), `inventory_log`, `macro_daily` (EVDS). FK ile bağlı. (Detay: bir önceki sürümde §3.1.)

---

## 6. Lisans/Erişim Uyarıları (özet)

- **Kaggle setleri:** İndirmeden önce her birinin "License" alanını kontrol et — sayfalar JS render olduğu için araştırmada otomatik teyit edilemedi. PaySim (CC BY-SA 4.0) ve Sparkov (CC0) net; HighRadius/IBM/Retail/DataCo lisansları belirsiz → bootcamp için kullan, "satın alma/dışarıdan hazır ürün" diskalifiye kuralına takılmaz (veri kullanımı ≠ hazır ürün), ama README'de kaynak+lisans belirt.
- **UK Payment Practices:** Open Government Licence — en temiz, ticari dahil serbest.
- **SDV/CTGAN:** Business Source License — bootcamp (ticari değil) için sorun yok.
- **Gretel:** Arşivlendi + source-available — kullanma.

---

## 7. Kaynaklar (doğrulanmış)

**Fatura / AR**
- [Payment Date Prediction for Invoices (pradumn203)](https://www.kaggle.com/datasets/pradumn203/payment-date-prediction-for-invoices-dataset) ✅
- [Payment Date Dataset – HighRadius (rajattomar132)](https://www.kaggle.com/datasets/rajattomar132/payment-date-dataset) ⚠️
- [Finance Factoring – IBM (hhenry)](https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories) ✅
- [UK Payment Practices Reporting (GOV.UK)](https://check-payment-practices.service.gov.uk/search/) ✅ OGL
- [Predicting Account Receivables with ML (arXiv 2008.07363)](https://arxiv.org/abs/2008.07363)
- [Optimize Cash Collection (arXiv 1912.10828)](https://arxiv.org/abs/1912.10828)
- [EU Payment Observatory](https://single-market-economy.ec.europa.eu/smes/challenges-and-resilience/late-payment/eu-payment-observatory_en)

**Nakit akışı (metodoloji)**
- [SME Financial Management System – cold-start (arXiv 2511.03631)](https://arxiv.org/pdf/2511.03631) ⭐
- [MLP/LSTM vs ARIMA/Prophet (Springer)](https://link.springer.com/article/10.1007/s10660-019-09362-7)
- [Working Capital ML review (AJSRI)](https://researchinnovationjournal.com/index.php/AJSRI/article/view/111)
- [Multi-step ARIMA/LightGBM/Prophet (TDS)](https://towardsdatascience.com/multi-step-time-series-forecasting-with-arima-lightgbm-and-prophet-cc9e3f95dfb0/)

**Stok / Tedarik**
- [Retail Store Inventory Forecasting (anirudhchauhan)](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) ✅
- [DataCo Smart Supply Chain (shashwatwork)](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) ✅
- [Store Sales – Favorita (Kaggle comp)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) ✅
- [Logistics & Supply Chain (datasetengineer)](https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset) ✅
- [1M Turkish Market Sales (omercolakoglu)](https://www.kaggle.com/datasets/omercolakoglu/1m-rows-turkish-market-sales-dataset) 🇹🇷
- [Customer Shopping Istanbul (mehmettahiraslan)](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset) 🇹🇷

**Banka / Anomali**
- [PaySim (ealaxi)](https://www.kaggle.com/datasets/ealaxi/paysim1) ✅
- [Sparkov Generator (namebrandon)](https://github.com/namebrandon/Sparkov_Data_Generation) ✅
- [Sparkov Dataset (kartik2112)](https://www.kaggle.com/datasets/kartik2112/fraud-detection) ✅
- [Credit Card Fraud (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ✅
- [Bank Statement NLP (apoorvwatsky)](https://www.kaggle.com/datasets/apoorvwatsky/bank-transaction-data)

**Makro (TCMB)**
- [evds (PyPI)](https://pypi.org/project/evds/) ✅ · [evdspy](https://pypi.org/project/evdspy/) · [borsapy](https://github.com/saidsurucu/borsapy) ✅ · [tcmb-py](https://github.com/kaymal/tcmb-py)
- [EVDS portal](https://evds3.tcmb.gov.tr/) · [API key + header değişikliği rehberi](https://urazakgul.github.io/python-blog/posts/post_9/)
- [yfinance](https://pypi.org/project/yfinance/) · [wbgapi](https://pypi.org/project/wbgapi/)

**Sentetik üretim**
- [Faker](https://github.com/joke2k/faker) ✅ · [Faker finansal işlem örneği](https://python-fiddle.com/examples/faker-financial-transactions) · [DataCamp tutorial](https://www.datacamp.com/tutorial/creating-synthetic-data-with-python-faker-tutorial)
- [SDV (ilişkisel)](https://docs.sdv.dev/sdv) ✅ · [CTGAN](https://github.com/sdv-dev/CTGAN) · [Mimesis](https://mimesis.name/master/about.html)

**Agent / RAG**
- [wyne1/financial-analyst-crewai](https://github.com/wyne1/financial-analyst-crewai) ✅ · [botextractai/ai-crewai-multi-agent](https://github.com/botextractai/ai-crewai-multi-agent) ✅
- [CrewAI Memory](https://docs.crewai.com/en/concepts/memory) ⚠️ · [Mem0 for CrewAI](https://mem0.ai/blog/crewai-memory-production-setup-with-mem0)
