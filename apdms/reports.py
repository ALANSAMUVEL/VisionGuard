"""PDF session report generation using ReportLab and Matplotlib."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger("APDMS.reports")

_STATE_PALETTE = {
    "ALERT": (0.18, 0.80, 0.18),
    "MILD": (0.90, 0.85, 0.10),
    "DROWSY": (0.95, 0.55, 0.10),
    "CRITICAL": (0.90, 0.15, 0.15),
}


class PDFReportGenerator:
    """Compiles completed session records into an analytical PDF summary report."""

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        driver_name: str,
        session_logs: List[Dict[str, Any]],
        session_id: Optional[Union[int, str]] = None,
    ) -> Optional[str]:
        """Generate PDF report for the completed session."""
        if not session_logs:
            logger.warning("Empty session logs; skipping PDF generation.")
            return None

        sid = str(session_id if session_id is not None else session_logs[0].get("session_id", "0"))
        safe_name = "".join(c for c in driver_name if c.isalnum() or c in ("-", "_")).strip() or "driver"
        pdf_filename = self.output_dir / f"session_{safe_name}_{sid}.pdf"

        try:
            self._build_pdf(str(pdf_filename), driver_name, sid, session_logs)
            return str(pdf_filename.resolve())
        except Exception as exc:
            logger.exception("Failed to build PDF report: %s", exc)
            return None

    def _build_pdf(
        self,
        filename: str,
        driver_name: str,
        session_id: str,
        logs: List[Dict[str, Any]],
    ) -> None:
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=0.6 * inch,
            leftMargin=0.6 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            alignment=1,
        )
        heading_style = ParagraphStyle(
            "HeadingStyle",
            parent=styles["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=4,
        )

        timestamps = [float(r.get("timestamp", 0.0)) for r in logs]
        t0 = timestamps[0] if timestamps else 0.0
        times = [t - t0 for t in timestamps]

        ears = [float(r.get("ear", 0.0) or 0.0) for r in logs]
        mars = [float(r.get("mar", 0.0) or 0.0) for r in logs]
        scores = [float(r.get("fatigue_score", 0.0) or 0.0) for r in logs]
        pitches = [float(r.get("pitch", 0.0) or 0.0) for r in logs]
        rolls = [float(r.get("roll", 0.0) or 0.0) for r in logs]
        yaws = [float(r.get("yaw", 0.0) or 0.0) for r in logs]
        states = [str(r.get("fsm_state", "ALERT") or "ALERT") for r in logs]
        perclos_vals = [float(r.get("perclos", 0.0) or 0.0) for r in logs]

        total_time = times[-1] if times else 0.0
        state_counts = {s: states.count(s) for s in ["ALERT", "MILD", "DROWSY", "CRITICAL"]}
        total_frames = max(1, len(states))

        alert_events = [r for r in logs if r.get("alert_event") not in (None, "", "null")]
        blink_counts = [int(r.get("blink_count", 0) or 0) for r in logs]
        yawn_counts = [int(r.get("yawn_count", 0) or 0) for r in logs]
        eye_closures = [float(r.get("eye_closure", 0.0) or 0.0) for r in logs]

        max_blinks = max(blink_counts) if blink_counts else 0
        max_yawns = max(yawn_counts) if yawn_counts else 0
        max_closure = max(eye_closures) if eye_closures else 0.0
        avg_ear = sum(ears) / len(ears) if ears else 0.0
        avg_mar = sum(mars) / len(mars) if mars else 0.0
        avg_perclos = sum(perclos_vals) / len(perclos_vals) if perclos_vals else 0.0
        max_score = max(scores) if scores else 0.0

        if state_counts.get("CRITICAL", 0) > 0:
            overall = "CRITICAL – Immediate rest required"
        elif state_counts.get("DROWSY", 0) / total_frames > 0.10:
            overall = "DROWSY – Consider taking a break"
        elif state_counts.get("MILD", 0) / total_frames > 0.20:
            overall = "MILD – Signs of driver fatigue observed"
        else:
            overall = "ALERT – Driver remained attentive"

        temp_files: List[str] = []
        g_ear_mar = self._plot_ear_mar(times, ears, mars, session_id)
        g_score = self._plot_fatigue_score(times, scores, session_id)
        g_head = self._plot_head_pose(times, pitches, rolls, yaws, session_id)
        g_state = self._plot_state_distribution(state_counts, session_id)

        for g in (g_ear_mar, g_score, g_head, g_state):
            if g:
                temp_files.append(g)

        elements = []
        elements.append(Paragraph("VisionGuard – Driver Monitoring Session Report", title_style))
        elements.append(Spacer(1, 4))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3A86FF")))
        elements.append(Spacer(1, 8))

        meta = [
            ["Driver Name", driver_name],
            ["Session Identifier", session_id],
            ["Session Duration", f"{total_time:.1f} s ({total_time / 60.0:.1f} min)"],
            ["Overall Status", overall],
        ]
        meta_table = Table(meta, colWidths=[1.6 * inch, 4.6 * inch])
        meta_table.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#3A86FF")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        img_w = 6.0 * inch
        img_h = 2.2 * inch

        if g_ear_mar:
            elements.append(Paragraph("Eye & Mouth Aspect Ratios Over Time", heading_style))
            elements.append(Image(g_ear_mar, width=img_w, height=img_h))
            elements.append(Spacer(1, 6))

        if g_score:
            elements.append(Paragraph("Fatigue Score Trajectory", heading_style))
            elements.append(Image(g_score, width=img_w, height=img_h))
            elements.append(Spacer(1, 6))

        if g_head:
            elements.append(Paragraph("Head Orientation (Pitch / Roll / Yaw)", heading_style))
            elements.append(Image(g_head, width=img_w, height=img_h))
            elements.append(Spacer(1, 6))

        if g_state:
            elements.append(Paragraph("Fatigue State Distribution", heading_style))
            elements.append(Image(g_state, width=3.0 * inch, height=2.2 * inch))
            elements.append(Spacer(1, 10))

        elements.append(Paragraph("Session Summary Statistics", heading_style))
        stats_data = [
            ["Metric", "Value"],
            ["Total Duration", f"{total_time:.1f} s ({total_time / 60.0:.1f} min)"],
            ["Max Fatigue Score", f"{max_score:.1f} / 100"],
            ["Average EAR", f"{avg_ear:.3f}"],
            ["Average MAR", f"{avg_mar:.3f}"],
            ["Average PERCLOS", f"{avg_perclos:.1f}%"],
            ["Total Blinks", str(max_blinks)],
            ["Total Yawns", str(max_yawns)],
            ["Max Eye Closure", f"{max_closure:.2f} s"],
            ["% Time ALERT", f"{(state_counts.get('ALERT', 0) / total_frames) * 100:.1f}%"],
            ["% Time MILD", f"{(state_counts.get('MILD', 0) / total_frames) * 100:.1f}%"],
            ["% Time DROWSY", f"{(state_counts.get('DROWSY', 0) / total_frames) * 100:.1f}%"],
            ["% Time CRITICAL", f"{(state_counts.get('CRITICAL', 0) / total_frames) * 100:.1f}%"],
            ["Alert Events Fired", str(len(alert_events))],
        ]
        stats_table = Table(stats_data, colWidths=[2.8 * inch, 3.4 * inch])
        stats_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3A86FF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F4FF"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(stats_table)

        if alert_events:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Alert Event History", heading_style))
            alert_data = [["Time (s)", "Event", "FSM State", "Score"]]
            for ev in alert_events:
                et = float(ev.get("timestamp", t0)) - t0
                alert_data.append([
                    f"{et:.1f}",
                    str(ev.get("alert_event", "")),
                    str(ev.get("fsm_state", "")),
                    f"{float(ev.get('fatigue_score', 0.0)):.1f}",
                ])
            alert_table = Table(alert_data, colWidths=[0.9 * inch, 2.7 * inch, 1.3 * inch, 1.3 * inch])
            alert_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FF6B6B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFF0F0"), colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ])
            )
            elements.append(alert_table)

        doc.build(elements)

        for temp in temp_files:
            try:
                os.remove(temp)
            except OSError:
                pass

    def _plot_ear_mar(
        self, times: List[float], ears: List[float], mars: List[float], session_id: str
    ) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(8, 2.6))
            ax.plot(times, ears, label="EAR", color="#3A86FF", linewidth=1.0)
            ax.plot(times, mars, label="MAR", color="#06D6A0", linewidth=1.0)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Ratio", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = str(self.output_dir / f"_tmp_ear_mar_{session_id}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            return path
        except Exception:
            return None

    def _plot_fatigue_score(
        self, times: List[float], scores: List[float], session_id: str
    ) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(8, 2.6))
            ax.fill_between(times, scores, alpha=0.15, color="#FF6B6B")
            ax.plot(times, scores, label="Fatigue Score", color="#FF6B6B", linewidth=1.0)
            ax.axhline(35, color="#FFD166", linestyle="--", linewidth=0.8, label="Mild (35)")
            ax.axhline(60, color="#F4A261", linestyle="--", linewidth=0.8, label="Drowsy (60)")
            ax.axhline(80, color="#E63946", linestyle="--", linewidth=0.8, label="Critical (80)")
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Score (0–100)", fontsize=8)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = str(self.output_dir / f"_tmp_score_{session_id}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            return path
        except Exception:
            return None

    def _plot_head_pose(
        self, times: List[float], pitches: List[float], rolls: List[float], yaws: List[float], session_id: str
    ) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(8, 2.6))
            ax.plot(times, pitches, label="Pitch", color="#8338EC", linewidth=0.9)
            ax.plot(times, rolls, label="Roll", color="#3A86FF", linewidth=0.9)
            ax.plot(times, yaws, label="Yaw", color="#FF006E", linewidth=0.9)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Degrees (°)", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = str(self.output_dir / f"_tmp_head_{session_id}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            return path
        except Exception:
            return None

    def _plot_state_distribution(
        self, state_counts: Dict[str, int], session_id: str
    ) -> Optional[str]:
        try:
            labels = []
            sizes = []
            colors_list = []
            for st, cnt in state_counts.items():
                if cnt > 0:
                    labels.append(st)
                    sizes.append(cnt)
                    colors_list.append(_STATE_PALETTE.get(st, (0.5, 0.5, 0.5)))
            if not sizes:
                return None
            fig, ax = plt.subplots(figsize=(4, 3.0))
            ax.pie(sizes, labels=labels, colors=colors_list, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
            fig.tight_layout()
            path = str(self.output_dir / f"_tmp_dist_{session_id}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            return path
        except Exception:
            return None


class PDFGenerator(PDFReportGenerator):
    """Backwards-compatibility alias."""
    pass
