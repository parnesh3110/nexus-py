"""Small AppKit helpers so every Nexus window looks like a native macOS app: SF Symbols, system
fonts and colors (they follow light/dark mode), System Settings style grouped boxes and rows,
real switches. Everything uses frames in points, origin bottom-left (AppKit's default)."""
from AppKit import (NSBox, NSButton, NSColor, NSFont, NSImage, NSImageView, NSSwitch, NSTextField, NSView)
from Foundation import NSMakeRect

LEFT, CENTER, RIGHT = 0, 2, 1          # NSTextAlignment on macOS
REGULAR, MEDIUM, SEMIBOLD, BOLD = 0.0, 0.23, 0.3, 0.4


def color(name: str, fallback: str = "labelColor"):
    """A system color by name, falling back on older macOS versions."""
    fn = getattr(NSColor, name, None) or getattr(NSColor, fallback)
    return fn()


def symbol_image(name: str, size: float = 16, weight: str = "regular", fallback: str = "circle"):
    from AppKit import NSImageSymbolConfiguration
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None) or \
        NSImage.imageWithSystemSymbolName_accessibilityDescription_(fallback, None)
    weights = {"regular": 0.0, "medium": 0.23, "semibold": 0.3, "bold": 0.4}
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, weights.get(weight, 0.0))
    return img.imageWithSymbolConfiguration_(cfg) if img is not None else None


def symbol(parent, name: str, x, y, box: float, size: float = 16, tint=None, weight: str = "regular"):
    iv = NSImageView.alloc().initWithFrame_(NSMakeRect(x, y, box, box))
    iv.setImage_(symbol_image(name, size, weight))
    if tint is not None:
        iv.setContentTintColor_(tint)
    parent.addSubview_(iv)
    return iv


def set_symbol(iv, name: str, size: float = 16, tint=None, weight: str = "regular"):
    iv.setImage_(symbol_image(name, size, weight))
    if tint is not None:
        iv.setContentTintColor_(tint)


def label(parent, text: str, x, y, w, h, size: float = 13, weight: float = REGULAR, fg=None, align=LEFT,
          wrap: bool = True):
    f = NSTextField.wrappingLabelWithString_(text) if wrap else NSTextField.labelWithString_(text)
    f.setFrame_(NSMakeRect(x, y, w, h))
    f.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
    f.setAlignment_(align)
    f.setTextColor_(fg if fg is not None else color("labelColor"))
    f.setSelectable_(False)
    parent.addSubview_(f)
    return f


def group(parent, x, y, w, h) -> NSView:
    """A System Settings style rounded box. Returns the view to put rows in."""
    box = NSBox.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    box.setBoxType_(4)             # NSBoxCustom
    box.setTitlePosition_(0)       # no title
    box.setCornerRadius_(10)
    box.setBorderWidth_(0.5)
    box.setBorderColor_(color("separatorColor"))
    box.setFillColor_(color("quaternarySystemFillColor", "controlBackgroundColor"))
    box.setContentViewMargins_((0, 0))
    parent.addSubview_(box)
    return box.contentView()


def separator(parent, x, y, w):
    line = NSBox.alloc().initWithFrame_(NSMakeRect(x, y, w, 1))
    line.setBoxType_(2)            # NSBoxSeparator
    parent.addSubview_(line)
    return line


def row(content: NSView, index: int, count: int, height: float, icon: str, title: str, subtitle: str = "",
        icon_tint=None):
    """Row `index` (0 = top) of `count` in a group: SF Symbol, title, optional subtitle.
    Returns (y of the row's bottom, row height) so a trailing control can be placed."""
    total = content.frame().size.height
    w = content.frame().size.width
    y = total - (index + 1) * height
    symbol(content, icon, 14, y + (height - 26) / 2, 26, 17, icon_tint or color("controlAccentColor", "systemBlueColor"))
    if subtitle:
        label(content, title, 52, y + height / 2 - 1, w - 200, 20, 13, MEDIUM, wrap=False)
        label(content, subtitle, 52, y + 6, w - 200, height / 2 - 6, 11, REGULAR, color("secondaryLabelColor"))
    else:
        label(content, title, 52, y + (height - 20) / 2, w - 200, 20, 13, MEDIUM, wrap=False)
    if index < count - 1:
        separator(content, 52, y, w - 52)
    return y, height


def switch(parent, x, y, on: bool, target, tag: int):
    s = NSSwitch.alloc().initWithFrame_(NSMakeRect(x, y, 42, 24))
    s.setState_(1 if on else 0)
    s.setTarget_(target)
    s.setAction_("clicked:")
    s.setTag_(tag)
    parent.addSubview_(s)
    return s


def push_button(parent, title: str, x, y, w, target, tag: int, primary: bool = False):
    b = NSButton.buttonWithTitle_target_action_(title, target, "clicked:")
    b.setFrame_(NSMakeRect(x, y, w, 32))
    b.setTag_(tag)
    b.setBezelStyle_(1)  # rounded push button
    if primary:
        b.setKeyEquivalent_("\r")  # default button: blue, responds to Return
    parent.addSubview_(b)
    return b
