import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import string

def generate_user_id():
    """
    Generate a user ID in format like 'A2XJ7L4S5RX4F7' (14 characters total)
    Always starts with 'A' followed by 13 random alphanumeric characters
    """
    # First character is always 'A'
    user_id = 'A'
    
    # Generate 13 random characters (letters and numbers)
    characters = string.ascii_uppercase + string.digits
    for _ in range(13):
        user_id += random.choice(characters)
    
    return user_id
def generate_training_data():
    """
    Generate 3 Excel files with 500 bot entries and 500 human entries each
    """
    products = ["Laptop", "Jeans", "Decoration Lamp"]
    
    # IP pools for bots (shared IPs)
    bot_ip_pool = [
        "203.0.113.5", "198.51.100.10", "192.0.2.15", 
        "203.0.113.20", "198.51.100.25", "192.0.2.30",
        "203.0.113.35", "198.51.100.40", "192.0.2.45"
    ]
    
    # Payment methods
    human_payment_methods = ["COD", "UPI", "Debit Card", "Credit Card", "Internet Banking"]
    bot_payment_methods = ["Credit Card", "Internet Banking"]
    device_types = ["Desktop", "Mobile", "Tablet"]
    
    for product_name in products:
        print(f"Generating training data for {product_name}...")
        
        all_entries = []
        
        # Generate 500 Human entries
        print(f"  Generating 500 human entries...")
        for i in range(500):
            # Generate random datetime within last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            base_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Human IP addresses (more diverse)
            human_ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            
            # Human UserID (format: A + 13 random chars)
            human_user_id = generate_user_id()
            
            human_entry = {
                "DateTime": base_time.strftime("%Y-%m-%d %H:%M:%S"),
                "BuyingTime_Seconds": random.randint(60, 300),  # 60-300 seconds
                "PageViewDuration": random.randint(30, 120),    # 30-120 seconds
                "CartTime_Seconds": random.randint(10, 60),     # 10-60 seconds
                "IP_Address": human_ip,
                "UserID": human_user_id,
                "CouponUsed": random.choice(["Yes", "No"]),     # Sometimes
                "DiscountApplied": random.choice(["Yes", "No"]), # Mixed
                "PaymentMethod": random.choice(human_payment_methods), # Varied (COD, UPI, Cards, Banking)
                "ProductViewCount": random.randint(3, 10),      # 3-10 views
                "ProductSearchCount": random.randint(1, 5),     # 1-5 searches
                "AddToCart_RemoveCount": random.randint(1, 3),  # 1-3 changes
                "ReviewsRead": random.randint(1, 10),           # 1-10 reviews
                "DeviceType": random.choice(device_types),      # Mixed devices
                "MouseClicks": random.randint(10, 50),          # 10-50 clicks
                "KeyboardStrokes": random.randint(20, 100),     # 20-100 strokes
                "ProductID": product_name,
                "Label": "human"
            }
            all_entries.append(human_entry)
        
        # Generate 500 Bot entries
        print(f"  Generating 500 bot entries...")
        for i in range(500):
            # Generate random datetime within last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            base_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Bot IP addresses (shared from pool)
            bot_ip = random.choice(bot_ip_pool)
            
            # Bot UserID (format: A + 13 random chars) 
            bot_user_id = generate_user_id()
            
            bot_entry = {
                "DateTime": base_time.strftime("%Y-%m-%d %H:%M:%S"),
                "BuyingTime_Seconds": random.randint(1, 15),    # 1-15 seconds
                "PageViewDuration": random.randint(0, 5),       # 0-5 seconds
                "CartTime_Seconds": random.randint(0, 3),       # 0-3 seconds
                "IP_Address": bot_ip,
                "UserID": bot_user_id,
                "CouponUsed": "Yes",                            # Always
                "DiscountApplied": "Yes",                       # Always
                "PaymentMethod": random.choice(bot_payment_methods), # Limited to Credit Card/Internet Banking
                "ProductViewCount": random.randint(0, 1),       # 0-1 views
                "ProductSearchCount": 0,                        # 0 searches
                "AddToCart_RemoveCount": 0,                     # 0 changes
                "ReviewsRead": 0,                               # 0 reviews
                "DeviceType": "Desktop",                        # Mostly desktop
                "MouseClicks": random.randint(3, 8),            # 3-8 clicks
                "KeyboardStrokes": random.randint(0, 5),        # 0-5 strokes
                "ProductID": product_name,
                "Label": "bot"
            }
            all_entries.append(bot_entry)
        
        # Shuffle the entries to mix humans and bots
        random.shuffle(all_entries)
        
        # Create DataFrame
        df = pd.DataFrame(all_entries)
        
        # Save to Excel file
        filename = f"training_{product_name.lower().replace(' ', '_')}.xlsx"
        df.to_excel(filename, index=False)
        
        print(f"  ✅ Created {filename} with {len(all_entries)} entries")
        print(f"     - Human entries: {len([e for e in all_entries if e['Label'] == 'human'])}")
        print(f"     - Bot entries: {len([e for e in all_entries if e['Label'] == 'bot'])}")
        print()

def display_sample_data():
    """
    Display sample data from the first training file for verification
    """
    try:
        # Read the first training file
        df = pd.read_excel("training_laptop.xlsx")
        
        print("="*80)
        print("SAMPLE DATA FROM TRAINING_LAPTOP.XLSX")
        print("="*80)
        
        # Show first 5 human entries
        human_samples = df[df['Label'] == 'human'].head(3)
        print("\n🧍 HUMAN BEHAVIOR SAMPLES:")
        print("-" * 50)
        for idx, row in human_samples.iterrows():
            print(f"UserID: {row['UserID']}")
            print(f"BuyingTime: {row['BuyingTime_Seconds']}s | PageView: {row['PageViewDuration']}s | CartTime: {row['CartTime_Seconds']}s")
            print(f"ProductViews: {row['ProductViewCount']} | Searches: {row['ProductSearchCount']} | Reviews: {row['ReviewsRead']}")
            print(f"MouseClicks: {row['MouseClicks']} | KeyStrokes: {row['KeyboardStrokes']} | Device: {row['DeviceType']}")
            print(f"IP: {row['IP_Address']} | Coupon: {row['CouponUsed']} | Payment: {row['PaymentMethod']}")
            print("-" * 50)
        
        # Show first 5 bot entries
        bot_samples = df[df['Label'] == 'bot'].head(3)
        print("\n🤖 BOT BEHAVIOR SAMPLES:")
        print("-" * 50)
        for idx, row in bot_samples.iterrows():
            print(f"UserID: {row['UserID']}")
            print(f"BuyingTime: {row['BuyingTime_Seconds']}s | PageView: {row['PageViewDuration']}s | CartTime: {row['CartTime_Seconds']}s")
            print(f"ProductViews: {row['ProductViewCount']} | Searches: {row['ProductSearchCount']} | Reviews: {row['ReviewsRead']}")
            print(f"MouseClicks: {row['MouseClicks']} | KeyStrokes: {row['KeyboardStrokes']} | Device: {row['DeviceType']}")
            print(f"IP: {row['IP_Address']} | Coupon: {row['CouponUsed']} | Payment: {row['PaymentMethod']}")
            print("-" * 50)
        
        print(f"\nTotal entries: {len(df)}")
        print(f"Human entries: {len(df[df['Label'] == 'human'])}")
        print(f"Bot entries: {len(df[df['Label'] == 'bot'])}")
        
    except FileNotFoundError:
        print("Training files not found. Please run generate_training_data() first.")

if __name__ == "__main__":
    print("🚀 Starting Training Data Generation...")
    print("="*60)
    
    # Generate training data
    generate_training_data()
    
    print("="*60)
    print("✅ Training data generation completed!")
    print("\nFiles created:")
    print("- training_laptop.xlsx")
    print("- training_jeans.xlsx") 
    print("- training_decoration_lamp.xlsx")
    print("\nEach file contains:")
    print("- 500 human behavior entries")
    print("- 500 bot behavior entries")
    print("- Total: 1000 entries per product")
    print("="*60)
    
    # Display sample data
    display_sample_data()