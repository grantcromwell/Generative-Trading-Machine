"""
Brainnet Blue GUI Launcher
Simple, idiot-proof tkinter interface for selecting trading symbols
"""

import tkinter as tk
from tkinter import font as tkfont
from typing import Optional


# Color scheme - BLUE theme
COLORS = {
    "bg_dark": "#0a1628",       # Deep navy background
    "bg_medium": "#1a2744",     # Medium blue for frames
    "bg_light": "#2a3f5f",      # Lighter blue for hover
    "accent": "#00d4ff",        # Cyan accent
    "accent_hover": "#00a8cc",  # Darker cyan on hover
    "text": "#ffffff",          # White text
    "text_dim": "#8899aa",      # Dimmed text
    "success": "#00ff88",       # Green for positive
    "danger": "#ff4466",        # Red for exit
}

# Market symbols mapping
MARKETS = {
    "BTC": {"symbol": "BTC-USD", "name": "Bitcoin", "color": "#f7931a"},
    "ETH": {"symbol": "ETH-USD", "name": "Ethereum", "color": "#627eea"},
    "ES": {"symbol": "ES=F", "name": "S&P 500 Futures", "color": "#00d4ff"},
    "NQ": {"symbol": "NQ=F", "name": "Nasdaq Futures", "color": "#00ff88"},
}


class BrainnetLauncher:
    """Blue-themed GUI launcher for Brainnet trading system."""
    
    def __init__(self):
        self.selection: Optional[str] = None
        self.root = tk.Tk()
        self._setup_window()
        self._create_widgets()
    
    def _setup_window(self):
        """Configure the main window."""
        self.root.title("🧠 BRAINNET")
        self.root.geometry("500x600")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(False, False)
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"500x600+{x}+{y}")
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Header
        header_frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        header_frame.pack(pady=30)
        
        # Brain emoji + title
        title_label = tk.Label(
            header_frame,
            text="🧠 BRAINNET",
            font=("Helvetica", 36, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_dark"]
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Adaptive Quant Trading System",
            font=("Helvetica", 14),
            fg=COLORS["text_dim"],
            bg=COLORS["bg_dark"]
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Instruction
        instruction_label = tk.Label(
            self.root,
            text="SELECT A MARKET TO ANALYZE",
            font=("Helvetica", 12, "bold"),
            fg=COLORS["text"],
            bg=COLORS["bg_dark"]
        )
        instruction_label.pack(pady=(20, 30))
        
        # Market buttons container
        buttons_frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        buttons_frame.pack(pady=10)
        
        # Create market buttons
        for key, info in MARKETS.items():
            self._create_market_button(buttons_frame, key, info)
        
        # Spacer
        spacer = tk.Frame(self.root, bg=COLORS["bg_dark"], height=40)
        spacer.pack()
        
        # Exit button
        exit_frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        exit_frame.pack(pady=20)
        
        exit_btn = tk.Button(
            exit_frame,
            text="✕  EXIT",
            font=("Helvetica", 14, "bold"),
            fg=COLORS["text"],
            bg=COLORS["danger"],
            activebackground="#cc3355",
            activeforeground=COLORS["text"],
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
            command=self._on_exit
        )
        exit_btn.pack()
        
        # Footer
        footer_label = tk.Label(
            self.root,
            text="Powered by Phi-3.5-Mini • GAF Pattern Recognition",
            font=("Helvetica", 10),
            fg=COLORS["text_dim"],
            bg=COLORS["bg_dark"]
        )
        footer_label.pack(side="bottom", pady=20)
    
    def _create_market_button(self, parent: tk.Frame, key: str, info: dict):
        """Create a styled market selection button."""
        btn_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        btn_frame.pack(pady=8)
        
        btn = tk.Button(
            btn_frame,
            text=f"{key}",
            font=("Helvetica", 24, "bold"),
            fg=COLORS["text"],
            bg=COLORS["bg_medium"],
            activebackground=COLORS["bg_light"],
            activeforeground=COLORS["text"],
            width=12,
            height=2,
            relief="flat",
            cursor="hand2",
            command=lambda s=info["symbol"]: self._on_select(s)
        )
        btn.pack()
        
        # Subtext with full name
        name_label = tk.Label(
            btn_frame,
            text=info["name"],
            font=("Helvetica", 10),
            fg=COLORS["text_dim"],
            bg=COLORS["bg_dark"]
        )
        name_label.pack()
        
        # Hover effects
        def on_enter(e):
            btn.configure(bg=COLORS["bg_light"])
        
        def on_leave(e):
            btn.configure(bg=COLORS["bg_medium"])
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
    
    def _on_select(self, symbol: str):
        """Handle market selection."""
        self.selection = symbol
        self.root.destroy()
    
    def _on_exit(self):
        """Handle exit button."""
        self.selection = None
        self.root.destroy()
    
    def run(self) -> Optional[str]:
        """Run the GUI and return selected symbol."""
        self.root.mainloop()
        return self.selection


def launch_selector_gui() -> Optional[str]:
    """
    Launch the Brainnet market selector GUI.
    
    Returns:
        Selected symbol string (e.g., 'BTC-USD') or None if exited
    """
    launcher = BrainnetLauncher()
    return launcher.run()


if __name__ == "__main__":
    # Test the GUI directly
    selected = launch_selector_gui()
    if selected:
        print(f"Selected: {selected}")
    else:
        print("Exited without selection")

