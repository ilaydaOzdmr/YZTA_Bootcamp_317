import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process

# .env dosyasını yükle ve terminal hafızasındakileri EZ (override=True)
ROOT = __file__.rsplit(os.sep, 3)[0]
load_dotenv(os.path.join(ROOT, '.env'), override=True)

from agents import cfo_agent, tahsilat_agent, tedarik_agent, risk_orkestrator
from tasks import task_cfo, task_tahsilat, task_tedarik, task_cozum_raporu

resilience_crew = Crew(
    agents=[cfo_agent, tahsilat_agent, tedarik_agent, risk_orkestrator],
    tasks=[task_cfo, task_tahsilat, task_tedarik, task_cozum_raporu],
    process=Process.sequential, 
    verbose=True,
    max_rpm=10
)

if __name__ == "__main__":
    print("ResilienceOS Otonom Yapay Zeka Orkestrası Başlatılıyor...")
    print("=" * 60)
    
    # Yeni API Anahtarının okunup okunmadığını kontrol et (Güvenlik için sadece son 4 hanesini yazdırır)
    api_key = os.environ.get("GEMINI_API_KEY", "BULUNAMADI")
    print(f"Kullanılan API Anahtarı Sonu: ...{api_key[-4:]}")
    print("=" * 60)
    
    result = resilience_crew.kickoff()
    
    report_path = Path(ROOT) / "reports" / "agent_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(str(result), encoding="utf-8")

    print("=" * 60)
    print("NİHAİ YÖNETİM RAPORU:\n")
    print(result)
    print(f"\nRapor kaydedildi: {report_path}")