"""
=============================================================================
Group 8 — MCE 415: YOLOv10 Road Anomaly Detection
inference_app.py  |  Interactive Desktop Inference App
=============================================================================
ZERO extra UI dependencies — uses only Python's built-in Tkinter.

Requirements (inference only):
    pip install ultralytics opencv-python Pillow torch

Run:
    python inference_app.py

Weight files:
    Place your trained .pt files under a weights/ folder next to this script:

        weights/lr_0.001/best.pt
        weights/lr_0.01/best.pt   (etc.)

    The app auto-discovers all .pt files recursively.
=============================================================================
"""

# ── Stdlib (zero install required) ───────────────────────────────────────────
import os
import sys
import time
import threading
import traceback
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Scale, OptionMenu, StringVar, IntVar,
    DoubleVar, Text, Scrollbar, Canvas, filedialog, messagebox,
    HORIZONTAL, END, DISABLED, NORMAL, LEFT, RIGHT,
    TOP, BOTTOM, BOTH, X, Y, W, E, CENTER, WORD, ttk,
)
from tkinter import font as tkfont


# ── Dependency check — runs before anything else ─────────────────────────────
def _check_deps():
    missing = []
    for pkg, name in [("PIL", "Pillow"), ("cv2", "opencv-python"),
                      ("ultralytics", "ultralytics>=8.2.50"), ("torch", "torch")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)
    if missing:
        try:
            root = Tk(); root.withdraw()
            messagebox.showerror(
                "Missing Dependencies",
                "Please install the following and try again:\n\n"
                + "\n".join(f"  pip install {p}" for p in missing)
                + "\n\nOr run:  pip install -r requirements.txt"
            )
            root.destroy()
        except Exception:
            pass
        print("\n[ERROR] Missing packages:", ", ".join(missing))
        print("Run:  pip install -r requirements.txt\n")
        sys.exit(1)

_check_deps()

import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


# ── Constants ─────────────────────────────────────────────────────────────────
APP_TITLE    = "YOLOv10 — Road Anomaly Detection  |  Group 8 · MCE 415"
WIN_W        = 1120
WIN_H        = 730
IMG_PANEL_H  = 310
CANVAS_W     = 390

CLASS_NAMES  = ["Pothole", "Speedbump", "Crack"]
CLASS_COLORS = {
    "Pothole"  : (212,  82,   0),
    "Speedbump": ( 50, 160,   0),
    "Crack"    : (220, 100,   0),
}

# Palette
C_HEADER_BG  = "#1a56db"
C_HEADER_FG  = "#ffffff"
C_BG         = "#f1f5f9"
C_PANEL      = "#ffffff"
C_BORDER     = "#e2e8f0"
C_BTN_RUN    = "#1a56db"
C_BTN_CLR    = "#64748b"
C_BTN_FG     = "#ffffff"
C_ACCENT     = "#1a56db"
C_MUTED      = "#64748b"
C_OK         = "#15803d"
C_ERR        = "#dc2626"
C_WARN       = "#92400e"

SCRIPT_DIR  = Path(__file__).parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

LR_LABEL_MAP = [
    ("1e-05",  "LR = 0.00001"),
    ("1e-5",   "LR = 0.00001"),
    ("0.0001", "LR = 0.0001"),
    ("0.001",  "LR = 0.001"),
    ("0.01",   "LR = 0.01"),
    ("0.1",    "LR = 0.1"),
]


# ── Weight discovery ──────────────────────────────────────────────────────────
def discover_weights() -> dict:
    found = {}
    for d in [WEIGHTS_DIR, SCRIPT_DIR]:
        if not d.exists():
            continue
        for pt in sorted(d.rglob("*.pt")):
            label = None
            for pattern, readable in LR_LABEL_MAP:
                if pattern in str(pt):
                    label = f"{readable}  [{pt.stem}.pt]"
                    break
            if label is None:
                label = f"{pt.parent.name}/{pt.name}"
            if label not in found:
                found[label] = pt
    return found


# ── Image helpers ─────────────────────────────────────────────────────────────
def pil_to_tk(pil_img, max_w, max_h):
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)

def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Detection drawing ─────────────────────────────────────────────────────────
def draw_detections(img_bgr, result):
    annotated = img_bgr.copy()
    counts    = {c: 0 for c in CLASS_NAMES}
    confs     = []
    rows      = []

    if result.boxes is None or len(result.boxes) == 0:
        return annotated, counts, confs, rows

    h, w  = annotated.shape[:2]
    thick = max(2, int(min(h, w) / 280))

    for box in result.boxes:
        cls_id          = int(box.cls[0])
        conf_val        = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh          = x2 - x1, y2 - y1
        cls_name        = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        color           = CLASS_COLORS.get(cls_name, (128, 128, 128))

        counts[cls_name] = counts.get(cls_name, 0) + 1
        confs.append(conf_val)
        rows.append([cls_name, f"{conf_val:.3f}", f"({x1}, {y1})", f"{bw}×{bh} px"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        label       = f"{cls_name} {conf_val:.2f}"
        fscale      = max(0.42, min(w, h) / 1100)
        fthick      = max(1, thick - 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)
        ly          = max(y1 - 6, th + 8)
        cv2.rectangle(annotated, (x1, ly - th - 6), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    (255, 255, 255), fthick, cv2.LINE_AA)

    return annotated, counts, confs, rows


# ═════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═════════════════════════════════════════════════════════════════════════════
class App:

    def __init__(self, root: Tk):
        self.root          = root
        self.weight_opts   = discover_weights()
        self._model_cache  = {}
        self._current_img  = None    # BGR ndarray
        self._last_result  = None    # BGR ndarray (annotated)
        self._tk_input     = None    # PhotoImage refs (prevent GC)
        self._tk_output    = None

        self._setup_window()
        self._setup_fonts()
        self._setup_treeview_style()
        self._build_ui()
        self._populate_weights()
        self._show_placeholder(self.canvas_in,  "Click  Browse  to load an image")
        self._show_placeholder(self.canvas_out, "Detection result will appear here")

    # ── Window ────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title(APP_TITLE)
        self.root.resizable(False, False)
        self.root.configure(bg=C_BG)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{WIN_W}x{WIN_H}+{(sw-WIN_W)//2}+{(sh-WIN_H)//2}")

    def _setup_fonts(self):
        self.f_header  = tkfont.Font(family="Helvetica", size=13, weight="bold")
        self.f_section = tkfont.Font(family="Helvetica", size=8,  weight="bold")
        self.f_label   = tkfont.Font(family="Helvetica", size=9,  weight="bold")
        self.f_body    = tkfont.Font(family="Helvetica", size=9)
        self.f_btn     = tkfont.Font(family="Helvetica", size=10, weight="bold")
        self.f_mono    = tkfont.Font(family="Courier",   size=9)
        self.f_small   = tkfont.Font(family="Courier",   size=8)

    def _setup_treeview_style(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        font=("Courier", 9), rowheight=24,
                        background="white", fieldbackground="white",
                        foreground="#1e293b", borderwidth=0)
        style.configure("Treeview.Heading",
                        font=("Helvetica", 8, "bold"),
                        background="#f1f5f9", foreground=C_MUTED,
                        relief="flat", borderwidth=0)
        style.map("Treeview", background=[("selected", "#dbeafe")])

    # ── Full UI build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = Frame(self.root, bg=C_HEADER_BG, height=52)
        hdr.pack(fill=X, side=TOP)
        hdr.pack_propagate(False)
        Label(hdr, text="YOLOv10  Road Anomaly Detection",
              font=self.f_header, bg=C_HEADER_BG, fg=C_HEADER_FG
              ).pack(side=LEFT, padx=20, pady=14)
        Label(hdr, text="Group 8  ·  MCE 415  ·  Pothole | Speedbump | Crack",
              font=self.f_body, bg=C_HEADER_BG, fg="#bfdbfe"
              ).pack(side=RIGHT, padx=20)

        # Status bar (built before body so it packs at bottom)
        self._build_status_bar()

        # Body
        body = Frame(self.root, bg=C_BG)
        body.pack(fill=BOTH, expand=True, padx=10, pady=(8, 4))
        self._build_left_panel(body)
        self._build_right_panel(body)

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        left = Frame(parent, bg=C_PANEL, bd=0,
                     highlightthickness=1, highlightbackground=C_BORDER,
                     width=268)
        left.pack(side=LEFT, fill=Y, padx=(0, 8), pady=(0, 0))
        left.pack_propagate(False)

        p = dict(padx=14, pady=(0, 0))

        # Model section
        self._sep(left, "MODEL")
        Label(left, text="Weight File", font=self.f_label,
              bg=C_PANEL, fg=C_MUTED).pack(anchor=W, **p)
        self.var_weight = StringVar()
        self.dd_weight  = OptionMenu(left, self.var_weight, "")
        self.dd_weight.config(font=self.f_body, bg=C_PANEL, fg="#1e293b",
                              activebackground="#f1f5f9", relief="flat",
                              highlightthickness=1, highlightbackground=C_BORDER,
                              width=28, anchor=W)
        self.dd_weight["menu"].config(font=self.f_body)
        self.dd_weight.pack(fill=X, padx=14, pady=(2, 6))

        # Settings section
        self._sep(left, "DETECTION SETTINGS")

        Label(left, text="Confidence Threshold", font=self.f_label,
              bg=C_PANEL, fg=C_MUTED).pack(anchor=W, **p)
        self.var_conf  = DoubleVar(value=0.25)
        self.lbl_conf  = Label(left, text="0.25", font=self.f_mono,
                               bg=C_PANEL, fg=C_ACCENT)
        self.lbl_conf.pack(anchor=E, padx=14)
        Scale(left, variable=self.var_conf, from_=0.05, to=0.95,
              resolution=0.05, orient=HORIZONTAL, showvalue=False,
              bg=C_PANEL, troughcolor=C_BORDER, highlightthickness=0,
              length=238,
              command=lambda v: self.lbl_conf.config(text=f"{float(v):.2f}")
              ).pack(padx=14, pady=(0, 6))

        Label(left, text="IoU Threshold (NMS)", font=self.f_label,
              bg=C_PANEL, fg=C_MUTED).pack(anchor=W, **p)
        self.var_iou  = DoubleVar(value=0.45)
        self.lbl_iou  = Label(left, text="0.45", font=self.f_mono,
                              bg=C_PANEL, fg=C_ACCENT)
        self.lbl_iou.pack(anchor=E, padx=14)
        Scale(left, variable=self.var_iou, from_=0.10, to=0.90,
              resolution=0.05, orient=HORIZONTAL, showvalue=False,
              bg=C_PANEL, troughcolor=C_BORDER, highlightthickness=0,
              length=238,
              command=lambda v: self.lbl_iou.config(text=f"{float(v):.2f}")
              ).pack(padx=14, pady=(0, 6))

        Label(left, text="Inference Image Size", font=self.f_label,
              bg=C_PANEL, fg=C_MUTED).pack(anchor=W, **p)
        self.var_imgsz = IntVar(value=640)
        sz_row = Frame(left, bg=C_PANEL)
        sz_row.pack(fill=X, padx=14, pady=(2, 8))
        for sz in [320, 480, 640, 800]:
            ttk.Radiobutton(sz_row, text=str(sz),
                            variable=self.var_imgsz, value=sz
                            ).pack(side=LEFT, padx=(0, 8))

        # Actions section
        self._sep(left, "ACTIONS")

        self.btn_browse = self._btn(left, "📂  Browse Image…",
                                    "#f1f5f9", "#1e293b", self._browse_image)
        self.btn_run    = self._btn(left, "▶  Run Detection",
                                    C_BTN_RUN, C_BTN_FG,
                                    self._run_inference_threaded, pady=8)
        self.btn_run.config(state=DISABLED)

        self.btn_save   = self._btn(left, "💾  Save Result",
                                    "#f1f5f9", "#1e293b", self._save_result)
        self.btn_save.config(state=DISABLED)

        self.btn_clear  = self._btn(left, "✕  Clear",
                                    C_BTN_CLR, C_BTN_FG, self._clear)

        # Summary section
        self._sep(left, "RUN SUMMARY")
        self.txt_summary = Text(
            left, font=self.f_small, bg="#f8fafc", fg="#334155",
            relief="flat", bd=0, wrap=WORD,
            highlightthickness=1, highlightbackground=C_BORDER,
            state=DISABLED, height=9,
        )
        self.txt_summary.pack(fill=X, padx=14, pady=(2, 8))

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        right = Frame(parent, bg=C_BG)
        right.pack(side=LEFT, fill=BOTH, expand=True)

        # ── Image canvases ─────────────────────────────────
        img_row = Frame(right, bg=C_BG)
        img_row.pack(fill=X, pady=(0, 8))

        def make_canvas_panel(parent, title):
            wrap = Frame(parent, bg=C_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=C_BORDER)
            wrap.pack(side=LEFT, fill=BOTH, expand=True,
                      padx=(0, 6) if title == "Input Image" else (0, 0))
            Label(wrap, text=title, font=self.f_section,
                  bg=C_PANEL, fg=C_MUTED).pack(anchor=W, padx=8, pady=(6, 2))
            c = Canvas(wrap, bg="#e2e8f0", bd=0,
                       highlightthickness=0, height=IMG_PANEL_H)
            c.pack(fill=BOTH, expand=True)
            return c

        self.canvas_in  = make_canvas_panel(img_row, "Input Image")
        self.canvas_out = make_canvas_panel(img_row, "Detection Result")

        # ── Detection table ────────────────────────────────
        tbl_wrap = Frame(right, bg=C_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=C_BORDER)
        tbl_wrap.pack(fill=BOTH, expand=True)

        tbl_hdr = Frame(tbl_wrap, bg=C_PANEL)
        tbl_hdr.pack(fill=X, padx=10, pady=(8, 4))
        Label(tbl_hdr, text="DETECTION LOG",
              font=self.f_section, bg=C_PANEL, fg=C_MUTED).pack(side=LEFT)
        self.lbl_det_count = Label(tbl_hdr, text="",
                                   font=self.f_section, bg=C_PANEL, fg=C_ACCENT)
        self.lbl_det_count.pack(side=RIGHT)

        cols = ("Class", "Confidence", "Top-Left (x, y)", "Size")
        self.tree = ttk.Treeview(tbl_wrap, columns=cols,
                                 show="headings", height=7,
                                 selectmode="browse")
        for col, w in zip(cols, [130, 110, 150, 110]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor=CENTER, stretch=False)
        self.tree.tag_configure("odd",  background="#ffffff")
        self.tree.tag_configure("even", background="#f8fafc")

        vsb = Scrollbar(tbl_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=RIGHT, fill=Y, padx=(0, 4), pady=4)
        self.tree.pack(fill=BOTH, expand=True, padx=(6, 0), pady=(0, 6))

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        Frame(self.root, bg=C_BORDER, height=1).pack(fill=X, side=BOTTOM)
        bar = Frame(self.root, bg="#f8fafc", height=28)
        bar.pack(fill=X, side=BOTTOM)
        bar.pack_propagate(False)
        self.lbl_status = Label(bar, text="Ready — browse an image to begin",
                                font=self.f_mono, bg="#f8fafc",
                                fg=C_MUTED, anchor=W)
        self.lbl_status.pack(side=LEFT, padx=12, fill=Y)
        dev = (f"GPU: {torch.cuda.get_device_name(0)}"
               if torch.cuda.is_available() else "CPU only")
        Label(bar, text=dev, font=self.f_mono,
              bg="#f8fafc", fg=C_MUTED).pack(side=RIGHT, padx=12)

    # ── UI helpers ────────────────────────────────────────────────────────────
    def _sep(self, parent, text):
        Frame(parent, bg=C_BORDER, height=1).pack(fill=X, padx=14, pady=(8, 3))
        Label(parent, text=text, font=self.f_section,
              bg=C_PANEL, fg=C_MUTED).pack(anchor=W, padx=14)

    def _btn(self, parent, text, bg, fg, cmd, pady=7):
        b = Button(parent, text=text, font=self.f_btn,
                   bg=bg, fg=fg, activebackground=bg,
                   relief="flat", cursor="hand2", pady=pady,
                   command=cmd)
        b.pack(fill=X, padx=14, pady=(2, 4))
        return b

    def _show_placeholder(self, canvas, text):
        canvas.update_idletasks()
        w = canvas.winfo_width()  or CANVAS_W
        h = canvas.winfo_height() or IMG_PANEL_H
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#e2e8f0", outline="")
        canvas.create_text(w // 2, h // 2, text=text,
                           font=self.f_body, fill="#94a3b8", justify=CENTER)

    def _set_status(self, text, color=C_MUTED):
        self.lbl_status.config(text=text, fg=color)
        self.root.update_idletasks()

    def _set_summary(self, text):
        self.txt_summary.config(state=NORMAL)
        self.txt_summary.delete("1.0", END)
        self.txt_summary.insert(END, text)
        self.txt_summary.config(state=DISABLED)

    def _display_on_canvas(self, canvas, img_bgr, tk_attr):
        canvas.update_idletasks()
        cw = canvas.winfo_width()  or CANVAS_W
        ch = canvas.winfo_height() or IMG_PANEL_H
        tk_img = pil_to_tk(bgr_to_pil(img_bgr), cw, ch)
        setattr(self, tk_attr, tk_img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=CENTER, image=tk_img)

    # ── Weight population ─────────────────────────────────────────────────────
    def _populate_weights(self):
        menu = self.dd_weight["menu"]
        menu.delete(0, END)
        if not self.weight_opts:
            self.var_weight.set("⚠  No weights found — see README")
            self.btn_run.config(state=DISABLED)
            self._set_status("No .pt weight files found. Place them in a weights/ folder.", C_ERR)
            return
        labels = list(self.weight_opts.keys())
        for lbl in labels:
            menu.add_command(label=lbl,
                             command=lambda l=lbl: self.var_weight.set(l))
        default = labels[0]
        for lbl in labels:
            if "0.001" in lbl and "0.0001" not in lbl:
                default = lbl
                break
        self.var_weight.set(default)
        self._set_status(f"{len(labels)} weight file(s) found. Browse an image to begin.")

    # ── Browse ────────────────────────────────────────────────────────────────
    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select a road image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._set_status(f"Could not read: {Path(path).name}", C_ERR)
            return
        self._current_img = img
        self._last_result = None
        self._show_placeholder(self.canvas_out, "Detection result will appear here")
        self._clear_table()
        self._set_summary("")
        self.btn_save.config(state=DISABLED)
        self._display_on_canvas(self.canvas_in, img, "_tk_input")
        if self.weight_opts:
            self.btn_run.config(state=NORMAL)
        h, w = img.shape[:2]
        self._set_status(
            f"Loaded: {Path(path).name}  ({w} × {h} px)  —  Click  Run Detection")

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_inference_threaded(self):
        self.btn_run.config(state=DISABLED)
        self.btn_browse.config(state=DISABLED)
        self._set_status("Running inference…", C_WARN)
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        try:
            self._inference_core()
        except Exception as e:
            msg = f"Error: {e}"
            self.root.after(0, lambda: self._set_status(msg, C_ERR))
            self.root.after(0, lambda: self._set_summary(traceback.format_exc()))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=NORMAL))
            self.root.after(0, lambda: self.btn_browse.config(state=NORMAL))

    def _inference_core(self):
        if self._current_img is None:
            self.root.after(0, lambda: self._set_status("No image loaded.", C_ERR))
            return

        label  = self.var_weight.get()
        if label not in self.weight_opts:
            self.root.after(0, lambda: self._set_status("Please select a valid model.", C_ERR))
            return

        wpath  = str(self.weight_opts[label])
        conf   = self.var_conf.get()
        iou    = self.var_iou.get()
        imgsz  = self.var_imgsz.get()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load / retrieve cached model
        self.root.after(0, lambda: self._set_status("Loading model…", C_WARN))
        if wpath not in self._model_cache:
            try:
                self._model_cache[wpath] = YOLO(wpath)
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Model load failed: {e}", C_ERR))
                return
        model = self._model_cache[wpath]

        self.root.after(0, lambda: self._set_status("Running inference…", C_WARN))

        t0      = time.perf_counter()
        results = model.predict(source=self._current_img, imgsz=imgsz,
                                conf=conf, iou=iou, device=device, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        result                        = results[0]
        annotated, counts, confs, rows = draw_detections(self._current_img, result)
        self._last_result              = annotated.copy()

        total    = sum(counts.values())
        fps      = 1000 / elapsed if elapsed > 0 else 0
        avg_conf = f"{sum(confs)/len(confs):.3f}" if confs else "—"
        hw       = (f"GPU ({torch.cuda.get_device_name(0)})"
                    if torch.cuda.is_available() else "CPU")

        summary = (
            f"Model  : {label}\n"
            f"Device : {hw}\n"
            f"ImgSz  : {imgsz} px\n"
            f"Conf   : {conf}   IoU : {iou}\n"
            f"Time   : {elapsed:.1f} ms\n"
            f"FPS    : {fps:.1f}\n"
            f"─────────────────────\n"
            f"Total  : {total} detection(s)\n"
        )
        for cls in CLASS_NAMES:
            summary += f"  {cls:<14}: {counts.get(cls, 0)}\n"
        summary += f"Avg conf : {avg_conf}"

        status = f"✓  {total} detection(s)  |  {elapsed:.1f} ms  |  {fps:.1f} FPS"

        self.root.after(0, lambda: self._display_on_canvas(
            self.canvas_out, annotated, "_tk_output"))
        self.root.after(0, lambda: self._populate_table(rows))
        self.root.after(0, lambda: self._set_summary(summary))
        self.root.after(0, lambda: self.lbl_det_count.config(
            text=f"{total} detection(s)"))
        self.root.after(0, lambda: self._set_status(status, C_OK))
        self.root.after(0, lambda: self.btn_save.config(state=NORMAL))

    # ── Table ─────────────────────────────────────────────────────────────────
    def _clear_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.lbl_det_count.config(text="")

    def _populate_table(self, rows):
        self._clear_table()
        for i, row in enumerate(rows):
            self.tree.insert("", END, values=row,
                             tags=("even" if i % 2 == 0 else "odd",))

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save_result(self):
        if self._last_result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
            initialfile="detection_result.jpg",
        )
        if not path:
            return
        try:
            cv2.imwrite(path, self._last_result)
            self._set_status(f"Saved → {Path(path).name}", C_OK)
        except Exception as e:
            self._set_status(f"Save failed: {e}", C_ERR)

    # ── Clear ─────────────────────────────────────────────────────────────────
    def _clear(self):
        self._current_img = self._last_result = None
        self._tk_input    = self._tk_output   = None
        self._show_placeholder(self.canvas_in,  "Click  Browse  to load an image")
        self._show_placeholder(self.canvas_out, "Detection result will appear here")
        self._clear_table()
        self._set_summary("")
        self.btn_run.config(state=DISABLED)
        self.btn_save.config(state=DISABLED)
        self._set_status("Cleared — browse an image to begin.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Group 8 — MCE 415  |  YOLOv10 Road Anomaly Detection")
    print("  Desktop Inference App  (Pure Tkinter — zero extra UI deps)")
    print("=" * 62)

    weights = discover_weights()
    if weights:
        print(f"\n  Found {len(weights)} weight file(s):")
        for lbl, pth in weights.items():
            print(f"    • {lbl}\n      → {pth}")
    else:
        print(f"\n  ⚠  No .pt files found. Expected: {WEIGHTS_DIR}")

    dev = ("CUDA — " + torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "CPU only")
    print(f"\n  Device : {dev}")
    print("  Launching UI …\n")

    root = Tk()
    App(root)
    root.mainloop()
