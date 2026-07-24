from crewai import Task
from agents import cfo_agent, tahsilat_agent, tedarik_agent, risk_orkestrator

task_cfo = Task(
    description='Nakit akışı aracını kullanarak önümüzdeki 30 gün içinde kasanın eksiye düşme olasılığını ve en olası kriz tarihini raporla.',
    expected_output='Kriz tarihi, minimum bakiye ve kriz olasılığını içeren kısa bir finansal özet.',
    agent=cfo_agent
)

task_tahsilat = Task(
    description='Riskli faturalar aracını kullanarak gecikmesi en muhtemel en büyük 3 faturayı listele.',
    expected_output='Fatura ID, Müşteri ID, Tutar ve tahmini gecikme günlerini içeren liste.',
    agent=tahsilat_agent
)

task_tedarik = Task(
    description='Stok tükenme riski aracını kullanarak operasyonu durdurabilecek en kritik ürünleri ve sipariş tarihlerini listele.',
    expected_output='Kritik ürünler, tükenme tarihleri ve maliyetleri içeren stok risk özeti.',
    agent=tedarik_agent
)

task_cozum_raporu = Task(
    description='Diğer ajanların bulgularını incele. Aksiyon önerileri aracını kullanarak mevcut kriz olasılığını sıfıra (%0) indirecek en iyi "Kombine Çözüm" paketini seç ve yönetime sunulacak nihai bir durum/aksiyon raporu yaz.',
    expected_output='Mevcut kriz durumu, departman bulguları ve seçilen en iyi aksiyon paketini içeren, profesyonel bir Yönetim Özeti (Markdown formatında).',
    agent=risk_orkestrator
)