"""
Menu system for fingerspell application.

Provides reusable menu component with keyboard navigation.
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
