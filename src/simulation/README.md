# Şok & Counterfactual Simülasyon Motoru

Nakit projeksiyonunun üzerine **aksiyon senaryoları** uygular ve her aksiyonun
**kriz olasılığını** ne kadar düşürdüğünü ölçer — rapordaki "Aksiyonları Uygula" motoru.

```bash
python src/models/train_invoice_delay.py
python src/models/forecast_cashflow.py
python src/simulation/shock_simulation.py
```

## Senaryolar (bugün = 31 Mayıs 2025, ufuk 30 gün, 2000 Monte Carlo yolu)

| Senaryo | Beklenen kasa | **Kriz olasılığı** | Durum |
|---------|--------------:|-------------------:|-------|
| Baseline (aksiyon yok) | +7.717 TL | **%73** | RISKLI |
| ABC erken odeme (%2 indirim) | +184.117 TL | **%0** | GUVENLI |
| Tedarikci odemesini 2 taksite bol (+10g) | +289.730 TL | **%0** | GUVENLI |
| Kur +%10 sok (downside) | -259.483 TL | **%100** | KRIZ |
| KOMBINE COZUM (1+2) | +466.130 TL | **%0** | GUVENLI |

**Kriz olasılığı = kasanın ufuk içinde eksiye düşme olasılığı** (Monte Carlo).

## Neden olasılık, neden sadece nokta tahmini değil?

Deterministik projeksiyon her faturayı **ortalama** tahmini tarihine koyar; bu, nakit
eğrisini düzleştirir ve **çukuru sistematik olarak sığlaştırır.**

Örnek: nokta tahmini **+7,717 TL** derken (kriz yok gibi),
Monte Carlo **%73 ihtimalle eksiye düşülür** diyor.
**Gerçekleşen: -59,780 TL** — olasılıksal tahmin doğru, nokta tahmini
yanıltıcıydı. Gerçek değer P5–P95 aralığının (-81,822 … 37,079)
içinde kalıyor.

### Belirsizlik nereden geliyor? (heteroskedastik σ)
Her fatura için gecikme, modelin kendi **out-of-fold artıklarından** çıkarılan dağılımdan
örneklenir. Saçılım tahmin değerine göre değişir: erken ödeyende ~2.5 gün, çok geç
ödeyende ~7.6 gün. **Tek bir sabit σ kullanmak riski küçümsüyordu**
(%60 → %73).

### Pazarlıklı ödemeler deterministiktir
Erken ödeme indirimi bir **sözleşmedir**, tahmin değil. Bu yüzden o faturaya model
gürültüsü eklenmez (`sigma_scale = 0`). Aksi halde "anlaştık ama yine de %20 ihtimalle
geç öder" gibi anlamsız bir sonuç çıkıyor ve aksiyon sıralaması bozuluyordu.

## Kritik tasarım notu: ölçüm penceresi

Erteleme süresi **tahmin penceresi içinde kalmalıdır**, aksi halde ertelenen taksit
pencerenin dışına düşer ve senaryo krizi *çözmüş gibi* görünür (ölçüm artefaktı).
`SUPPLIER_DEFER_DAYS = 10`: 17 Haziran + 10 = **27 Haziran** < 30 Haziran ✓
(tedarikçi sözleşmelerindeki esneklik payı 3–15 gün).

## Çıktılar
`reports/shock_scenarios.json` (kriz olasılıkları dahil) · `reports/shock_projection_curves.csv`
