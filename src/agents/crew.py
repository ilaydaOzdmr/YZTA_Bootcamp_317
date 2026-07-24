"""
ResilienceOS - CrewAI Orkestrasyon Giris Noktasi
Calistir:  python src/agents/crew.py
Cikti:     reports/agent_report.md
"""
import json
import time
from pathlib import Path
from crewai import Crew, Process
from .agents import cfo_ajan, tahsilat_ajan, tedarik_ajan, risk_denetcisi
from .tasks import nakit_gorevi, tahsilat_gorevi, tedarik_gorevi, orkestrasyon_gorevi

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def run() -> str:
    """Crew'u calistir, sonucu Markdown olarak kaydet ve dondur."""
    crew = Crew(
        agents=[cfo_ajan, tahsilat_ajan, tedarik_ajan, risk_denetcisi],
        tasks=[nakit_gorevi, tahsilat_gorevi, tedarik_gorevi, orkestrasyon_gorevi],
        process=Process.sequential,   # sirayla calis, donguden kac
        verbose=False,
    )

    # Gorevler arasi kisa bekleme: dakika basi istek sinirini korur
    original_kickoff = crew.kickoff

    def rate_limited_kickoff(**kwargs):
        result = original_kickoff(**kwargs)
        time.sleep(6)
        return result

    result = crew.kickoff()
    output = str(result)

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "agent_report.md"
    report_path.write_text(output, encoding="utf-8")
    print(f"\nRapor kaydedildi: {report_path}")
    return output


if __name__ == "__main__":
    print("ResilienceOS Ajan Orkestrasyonu basliyor...")
    print("Not: Gemini Flash (gemini-1.5-flash) kullaniliyor, max 10 istek/dakika\n")
    run()
