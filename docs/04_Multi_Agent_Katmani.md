# Multi-Agent Katman (CrewAI + Gemini)

> Sprint 3'te uygulandı. Bu doküman `src/agents/` altındaki **çalışan** ajan katmanını
> anlatır. Tasarım gerekçesi için: [`02_Sprint2_Agent_Plani.md`](02_Sprint2_Agent_Plani.md).

## Ne yapıyor

Sprint 1-2'de kurulan ML + simülasyon motorlarını (`twin_api`) okuyup **yorumlayan**,
birbirinin çıktısını değerlendirip **tek bir yönetim raporu** üreten 4 yapay zekâ ajanı.

**Önemli ayrım:** Sayılar deterministik motordan (`twin_api`) gelir; LLM **hesaplamaz**,
yorumlar ve sentezler. Yani "ajanlar krizi hesapladı" demiyoruz — ajanlar motorun çıktısını
okuyup aksiyon önerisine ve yönetici özetine çeviriyor.

## Mimari

```
twin_api (deterministik motor)
   │  cash_forecast / invoice_risk / stock_risk / recommend
   ▼
tools.py  (@tool sarmalayıcıları — JSON döner)
   ▼
4 ajan (agents.py) ──sıralı──> Crew (main_crew.py) ──> reports/agent_report.md
```

| Ajan | Rolü | Çağırdığı tool |
|------|------|----------------|
| **CFO Agent** | 30 günlük nakit darboğazını + kriz tarihini raporlar | `tool_cash_forecast` |
| **Tahsilat Agent** | Gecikme riski en yüksek 3 faturayı listeler | `tool_invoice_risk` |
| **Tedarik Zinciri Agent** | Kritik stok tükenme riskini analiz eder | `tool_stock_risk` |
| **Risk Orkestratör** | Diğer 3 ajanın bulgusunu birleştirip en iyi aksiyon paketini seçer, nihai raporu yazar | `tool_recommend` |

**Akış:** `Process.sequential` — ajanlar sırayla çalışır. Orkestratör görevine
`context=[task_cfo, task_tahsilat, task_tedarik]` verilir, yani diğer üç ajanın çıktısını
okuyarak karar verir.

## Teknik kararlar

- **LLM:** Gemini 2.0 Flash (`model="gemini/gemini-2.0-flash"`), `temperature=0.1`
  (odaklı, tutarlı cevaplar).
- **Rate limit:** `max_rpm=10` — dakikadaki istek sayısını sınırlar (429 hatasını azaltır).
- **max_iter=3:** her ajanın ReAct döngüsü üst sınırı.
- **Ortak hafıza (RAG):** başlangıç sürümünde kullanılmadı; ajanlar arası aktarım `context`
  ile yapılır. (ChromaDB/FAISS backlog'da opsiyonel.)

## Çalıştırma

```bash
# 1) Gemini API anahtarı — https://aistudio.google.com/api-keys (ücretsiz)
#    Proje kök dizininde .env dosyası:
#    GEMINI_API_KEY=...

# 2) Bağımlılıklar (crewai dahil)
pip install -r requirements.txt

# 3) Ön koşul: data/digital_twin.db + eğitilmiş modeller mevcut olmalı
#    (yoksa README'deki 'Hızlı Başlangıç' pipeline'ı çalıştırın)

# 4) Ajanları çalıştır
python src/agents/main_crew.py
```

Çıktı: terminale basılır **ve** `reports/agent_report.md` dosyasına kaydedilir (Markdown
yönetim özeti).

## Önemli notlar

- **Kota:** Bir koşu ≈ 10-16 LLM isteği (4 ajan × ReAct döngüsü). Gemini ücretsiz katmanının
  **günlük** sınırı vardır — geliştirirken crew'u tekrar tekrar çalıştırmak kotayı tüketir.
  Bir kez koşup çıktıyı (`agent_report.md`) kullanmak önerilir.
- **Demo/deploy:** Sayfa her açılışta LLM çağırmamalı — kaydedilen `agent_report.md`
  servis edilmeli (mevcut canlı/offline deseniyle aynı mantık). Aksi halde demo sırasında
  kota biterse ekran boş kalır.

## Dosyalar

| Dosya | İçerik |
|-------|--------|
| `src/agents/tools.py` | `twin_api` fonksiyonlarının `@tool` sarmalayıcıları |
| `src/agents/agents.py` | LLM tanımı + 4 ajan |
| `src/agents/tasks.py` | 4 görev (orkestratör `context` alır) |
| `src/agents/main_crew.py` | Crew kurulumu + çalıştırıcı + rapor kaydı |
