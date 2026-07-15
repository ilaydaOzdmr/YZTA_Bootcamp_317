# Şok & Counterfactual Simülasyon Motoru

Nakit projeksiyonunun üzerine **aksiyon ve şok senaryoları** uygular ve her birinin
**kriz olasılığını** (Monte Carlo) hesaplar — rapordaki "Aksiyonları Uygula" motoru.

```bash
python src/models/train_invoice_delay.py
python src/models/forecast_cashflow.py
python src/simulation/shock_simulation.py
```

## Baseline (aksiyon yok)
Beklenen kasa **+7.717 TL** · **kriz olasılığı %73** · RISKLI

## Aksiyonlar (kriz olasılığını düşürür)

| Aksiyon | Beklenen kasa | Kriz olasılığı | Durum |
|---------|--------------:|---------------:|-------|
| ABC erken odeme (%2 indirim) | +184.117 TL | **%0** | GUVENLI |
| Tahsilat kampanyasi (en riskli 3 fatura) | +193.157 TL | **%0** | GUVENLI |
| Buyuk faturayi faktore sat (%5 iskonto) | +178.717 TL | **%0** | GUVENLI |
| Tedarikci odemesini 2 taksite bol (+10g) | +289.730 TL | **%0** | GUVENLI |
| KOMBINE COZUM (erken odeme + tedarikci bolme) | +466.130 TL | **%0** | GUVENLI |

## Şoklar / downside (kriz olasılığını artırır)

| Şok | Beklenen kasa | Kriz olasılığı | Durum |
|-----|--------------:|---------------:|-------|
| SOK: genel tahsilat yavaslamasi (tum alacaklar +15g) | -894.786 TL | **%100** | KRIZ |
| SOK: kur +%10 (ithal odeme, EVDS'den) | -262.155 TL | **%100** | KRIZ |

**Kriz olasılığı = kasanın 30 günlük ufuk içinde eksiye düşme olasılığı** (2000 Monte Carlo yolu).

## Neden olasılık, neden sadece nokta tahmini değil?
Deterministik projeksiyon her faturayı **ortalama** tahmini tarihine koyar; çukuru
sistematik olarak sığlaştırır. Nokta tahmini +7.717 TL "kriz yok" derken,
Monte Carlo **%73 ihtimalle eksiye düşülür** diyor.
**Gerçekleşen: -59,780 TL** — dağılımın P5–P95 aralığında
(-81,822 … 37,079).

### Belirsizlik: heteroskedastik σ
Her fatura için gecikme, modelin kendi **out-of-fold artıklarından** çıkarılan dağılımdan
örneklenir. Saçılım tahmine göre değişir: erken ödeyende ~2.5 gün, çok geç ödeyende ~7.6 gün.
Sabit σ riski küçümsüyordu (%60 → %73).

### Pazarlıklı ödemeler deterministiktir
Erken ödeme / faktoring bir **sözleşmedir**, tahmin değil → o faturaya model gürültüsü
eklenmez (`sigma_scale = 0`).

## Aksiyon türleri
- **Erken ödeme indirimi (%2):** en büyük riskli alacağı vadesinde tahsil.
- **Tahsilat kampanyası:** en riskli 3 açık faturaya erken ödeme teşviki.
- **Faktoring:** büyük faturayı %5 iskontoyla anında nakde çevir (riski faktöre devret).
- **Tedarikçi bölme (+10g):** büyük ödemeyi 2 taksite yay (pencere içinde; toplam korunur).
- **Kombine:** erken ödeme + tedarikçi bölme.

## Şoklar
- **Genel tahsilat yavaşlaması:** tüm açık alacaklar +15 gün (resesyon senaryosu).
- **Kur şoku:** ithal ödeme artar; şok büyüklüğü **gerçek EVDS USD/TRY volatilitesinden**
  türetilir (sabit değil).

## Çıktılar
`reports/shock_scenarios.json` (kriz olasılıkları + is_action) · `reports/shock_projection_curves.csv`
