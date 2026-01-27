"""
Menu system for fingerspell application.

Provides reusable menu component with keyboard navigation.
Includes PaginatedMenu for dynamic content with pagination.
"""

import cv2
import numpy as np
from src.fingerspell.ui.common import draw_text


class Menu:
    """
    Reusable menu component for OpenCV windows.
    
    Displays title, options, and instructions using consistent styling.
    Handles keyboard input for selection.
    """
    
    def __init__(self, title, options, window_name="Menu", width=1280, height=720):
        """
        Initialize menu.
        
        Args:
            title: Menu title text
            options: List of tuples (key, description) e.g., [('1', 'Collect Data'), ...]
                    Use ('', 'text') for non-selectable items like blank lines or ESC
            window_name: OpenCV window name
            width: Window width
            height: Window height
        """
        self.title = title
        self.options = options
        self.window_name = window_name
        self.width = width
        self.height = height
        self.selected = None
    
    def draw(self, image):
        """
        Draw menu on image.
        
        Args:
            image: Black screen to draw on
            
        Returns:
            Modified image
        """
        h, w = image.shape[:2]
        
        # Draw title
        title_y = 100
        image = draw_text(image, self.title, (w//2 - 250, title_y),
                         font_size=42, color=(255, 255, 255))
        
        # Draw options
        option_y = 220
        for key, description in self.options:
            # Highlight if selected
            if key == self.selected:
                color = (0, 255, 255)  # Cyan
                font_size = 30
            else:
                color = (255, 255, 255)  # White
                font_size = 26
            
            text = f"{key} - {description}" if key else description
            image = draw_text(image, text, (w//2 - 200, option_y),
                             font_size=font_size, color=color)
            option_y += 60
        
        # Draw instructions at bottom
        instructions = "Press number key to select, ESC to quit"
        image = draw_text(image, instructions, (w//2 - 220, h - 80),
                         font_size=20, color=(200, 200, 200))
        
        return image
    
    def run(self):
        """
        Run the menu and wait for selection.
        
        Returns:
            str: Selected option key, or None if ESC pressed
        """
        # Create black screen
        screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        print(f"\n{self.title}")
        print("=" * 60)
        for key, description in self.options:
            if key:
                print(f"  {key} - {description}")
            else:
                print(f"  {description}")
        print("=" * 60)
        
        while True:
            display = screen.copy()
            display = self.draw(display)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            
            # Check if key matches any option
            key_char = chr(key) if 32 <= key <= 126 else None
            for option_key, _ in self.options:
                if option_key and key_char == option_key:
                    self.selected = option_key
                    cv2.destroyAllWindows()
                    return option_key


class PaginatedMenu(Menu):
    """
    Menu with pagination support for dynamic content.
    
    Displays numbered items (1-9, 0 for 10th) with n/p navigation.
    Accepts a formatting function to customize item display.
    """
    
    def __init__(self, title, items, format_fn=None, items_per_page=10, 
                 window_name="Menu", width=1280, height=720):
        """
        Initialize paginated menu.
        
        Args:
            title: Menu title text
            items: List of items to display
            format_fn: Function(item) -> (main_text, detail_text). 
                      If None, uses str(item) as main text
            items_per_page: Number of items per page (max 10)
            window_name: OpenCV window name
            width: Window width
            height: Window height
        """
        # Call parent init with empty options
        super().__init__(title, [], window_name, width, height)
        
        self.items = items
        self.format_fn = format_fn or self._default_format
        self.items_per_page = min(items_per_page, 10)  # Max 10 items (1-9, 0)
        self.current_page = 0
        self.total_pages = (len(items) + items_per_page - 1) // items_per_page if items else 0
    
    def _default_format(self, item):
        """Default formatting: just convert to string."""
        return (str(item), "")
    
    def get_current_items(self):
        """Get items for current page."""
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.items))
        return self.items[start_idx:end_idx]
    
    def draw(self, image):
        """
        Draw paginated menu on image.
        
        Args:
            image: Black screen to draw on
            
        Returns:
            Modified image
        """
        h, w = image.shape[:2]
        
        # Draw title
        title_y = 60
        image = draw_text(image, self.title, (50, title_y),
                         font_size=36, color=(255, 255, 255))
        
        # Draw separator
        cv2.line(image, (50, 90), (w - 50, 90), (200, 200, 200), 2)
        
        # Draw items
        page_items = self.get_current_items()
        y_pos = 140
        
        for i, item in enumerate(page_items, 1):
            # Item number (1-9, 0 for 10)
            display_num = i if i < 10 else 0
            
            # Get formatted text
            main_text, detail_text = self.format_fn(item)
            
            # Draw main text
            item_text = f"  {display_num} - {main_text}"
            image = draw_text(image, item_text, (80, y_pos),
                             font_size=22, color=(255, 255, 255))
            
            # Draw detail text if present
            if detail_text:
                image = draw_text(image, f"      {detail_text}", (80, y_pos + 28),
                                 font_size=16, color=(180, 180, 180))
            
            y_pos += 55
        
        # Draw pagination info
        if self.total_pages > 1:
            page_info = f"Page {self.current_page + 1}/{self.total_pages}"
            image = draw_text(image, page_info, (50, h - 60),
                             font_size=18, color=(200, 200, 200))
            
            nav_text = "n - Next Page | p - Previous Page"
            image = draw_text(image, nav_text, (w // 2 - 150, h - 60),
                             font_size=18, color=(200, 200, 200))
        
        # Draw instructions
        image = draw_text(image, "ESC - Cancel", (w - 200, h - 60),
                         font_size=18, color=(200, 200, 200))
        
        return image
    
    def run(self):
        """
        Run the paginated menu and wait for selection.
        
        Returns:
            Selected item, or None if cancelled
        """
        if not self.items:
            return None
        
        # Create black screen
        screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        while True:
            display = screen.copy()
            display = self.draw(display)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC - cancel
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('n'):  # Next page
                if self.current_page < self.total_pages - 1:
                    self.current_page += 1
            
            elif key == ord('p'):  # Previous page
                if self.current_page > 0:
                    self.current_page -= 1
            
            elif ord('0') <= key <= ord('9'):  # Number selection
                # Convert key to selection index
                if key == ord('0'):
                    selected_idx = 9  # 0 represents 10th item
                else:
                    selected_idx = key - ord('1')  # 1-9
                
                # Check if selection is valid for current page
                page_items = self.get_current_items()
                if 0 <= selected_idx < len(page_items):
                    cv2.destroyAllWindows()
                    return page_items[selected_idx]


# Smoke test
if __name__ == '__main__':
    print("Menu class smoke test")
    print("=" * 60)
    
    # Test menu
    menu = Menu(
        title="NGT Fingerspelling System",
        options=[
            ('1', 'Collect Training Data'),
            ('2', 'Train Models'),
            ('3', 'Run Recognition'),
            ('4', 'Alphabet Quiz'),
            ('', ''),  # Blank line
            ('', 'ESC - Quit')
        ],
        window_name="Test Menu"
    )
    
    print("\nShowing menu (press a number key or ESC)...")
    choice = menu.run()
    
    if choice:
        print(f"\nYou selected: {choice}")
    else:
        print("\nMenu cancelled")
    
    print("\nSmoke test complete!")
