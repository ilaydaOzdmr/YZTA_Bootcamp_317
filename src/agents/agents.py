import os
from crewai import Agent
from crewai import LLM
from tools import tool_cash_forecast, tool_invoice_risk, tool_stock_risk, tool_recommend

# Gemini 2.0 Flash modelini tanımlıyoruz
# CrewAI'ın kendi LLM sınıfını kullanarak Gemini'yi tanımlıyoruz
llm = LLM(
    model="gemini/gemini-2.0-flash", 
    api_key=os.environ.get("GEMINI_API_KEY"),
    max_rpm=10, # Crucial: Limits requests per minute to avoid 429 errors
    temperature=0.1 # Lower temperature makes responses more focused and concise
)

cfo_agent = Agent(
    role='Chief Financial Officer (CFO)',
    goal='Şirketin önündeki 30 günlük nakit darbogazlarını ve kriz tarihlerini tespit etmek.',
    backstory='Şirketin finansal sağlığından sorumlusun. Nakit projeksiyonlarını okuyup kriz ihtimallerini çok net bir şekilde analiz edersin.',
    tools=[tool_cash_forecast],
    llm=llm,
    verbose=True,
    max_iter=3
)

tahsilat_agent = Agent(
    role='Tahsilat ve Alacak Yöneticisi',
    goal='Gecikme ihtimali en yüksek faturaları tespit edip nakit girişini hızlandıracak stratejiler belirlemek.',
    backstory='Müşteri ödeme alışkanlıklarını analiz ederek hangi faturanın gecikeceğini bilir ve erken ödeme/tahsilat stratejileri üretirsin.',
    tools=[tool_invoice_risk],
    llm=llm,
    verbose=True,
    max_iter=3
)

tedarik_agent = Agent(
    role='Tedarik Zinciri Yöneticisi',
    goal='Kritik stok tükenme risklerini tespit edip operasyonun durmasını engellemek.',
    backstory='Hangi ham maddenin ne zaman tükeneceğini ve tedarikçilerin gecikme risklerini hesaplayarak zamanında sipariş planlaması yaparsın.',
    tools=[tool_stock_risk],
    llm=llm,
    verbose=True,
    max_iter=3
)

risk_orkestrator = Agent(
    role='Risk Denetçisi ve Strateji Orkestratörü',
    goal='Tüm departmanların verilerini ve simülasyon motorunun aksiyon önerilerini analiz ederek nihai bir Kriz Çözüm Raporu oluşturmak.',
    backstory='Sistemin bütünsel risklerini görürsün. Departmanlardan gelen verileri simülasyon motorunun aksiyonlarıyla birleştirip şirketi kurtaracak en mantıklı adımı (Örn: Erken ödeme indirimi + Tedarikçi taksitlendirme) seçersin.',
    tools=[tool_recommend],
    llm=llm,
    verbose=True,
    max_iter=3
)