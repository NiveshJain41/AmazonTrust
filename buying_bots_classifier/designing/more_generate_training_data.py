import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import string
import os

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

def generate_extended_training_data():
    """
    Generate 3 Excel files:
    - 10,000 correct bot entries
    - 10,000 correct human entries
    - 10 wrong bot entries (human-like behavior, but labeled 'bot') - UPDATED
    - 10 wrong human entries (bot-like behavior, but labeled 'human') - UPDATED
    for each product.
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
    
    # Define parameters for human-like and bot-like behavior
    human_params = {
        "BuyingTime_Seconds": (60, 300),
        "PageViewDuration": (30, 120),
        "CartTime_Seconds": (10, 60),
        "CouponUsed": ["Yes", "No"],
        "DiscountApplied": ["Yes", "No"],
        "PaymentMethod": human_payment_methods,
        "ProductViewCount": (3, 10),
        "ProductSearchCount": (1, 5),
        "AddToCart_RemoveCount": (1, 3),
        "ReviewsRead": (1, 10),
        "DeviceType": device_types,
        "MouseClicks": (10, 50),
        "KeyboardStrokes": (20, 100),
        "IP_Pool": None # Human IPs are diverse
    }

    bot_params = {
        "BuyingTime_Seconds": (1, 15),
        "PageViewDuration": (0, 5),
        "CartTime_Seconds": (0, 3),
        "CouponUsed": ["Yes"],
        "DiscountApplied": ["Yes"],
        "PaymentMethod": bot_payment_methods,
        "ProductViewCount": (0, 1),
        "ProductSearchCount": (0, 0),
        "AddToCart_RemoveCount": (0, 0),
        "ReviewsRead": (0, 0),
        "DeviceType": ["Desktop"],
        "MouseClicks": (3, 8),
        "KeyboardStrokes": (0, 5),
        "IP_Pool": bot_ip_pool
    }

    def create_entry(product_name, behavior_params, label):
        """Helper to create a single entry based on behavior parameters and label"""
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        base_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        entry = {
            "DateTime": base_time.strftime("%Y-%m-%d %H:%M:%S"),
            "BuyingTime_Seconds": random.randint(*behavior_params["BuyingTime_Seconds"]),
            "PageViewDuration": random.randint(*behavior_params["PageViewDuration"]),
            "CartTime_Seconds": random.randint(*behavior_params["CartTime_Seconds"]),
            "IP_Address": random.choice(behavior_params["IP_Pool"]) if behavior_params["IP_Pool"] else f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "UserID": generate_user_id(),
            "CouponUsed": random.choice(behavior_params["CouponUsed"]),
            "DiscountApplied": random.choice(behavior_params["DiscountApplied"]),
            "PaymentMethod": random.choice(behavior_params["PaymentMethod"]),
            "ProductViewCount": random.randint(*behavior_params["ProductViewCount"]),
            "ProductSearchCount": random.randint(*behavior_params["ProductSearchCount"]),
            "AddToCart_RemoveCount": random.randint(*behavior_params["AddToCart_RemoveCount"]),
            "ReviewsRead": random.randint(*behavior_params["ReviewsRead"]),
            "DeviceType": random.choice(behavior_params["DeviceType"]),
            "MouseClicks": random.randint(*behavior_params["MouseClicks"]),
            "KeyboardStrokes": random.randint(*behavior_params["KeyboardStrokes"]),
            "ProductID": product_name,
            "Label": label
        }
        return entry

    for product_name in products:
        print(f"Generating extended training data for {product_name}...")
        
        all_entries = []
        
        # Generate 10,000 Correct Human entries
        print(f"  Generating 10,000 correct human entries...")
        for _ in range(10000):
            all_entries.append(create_entry(product_name, human_params, "human"))
        
        # Generate 10,000 Correct Bot entries
        print(f"  Generating 10,000 correct bot entries...")
        for _ in range(10000):
            all_entries.append(create_entry(product_name, bot_params, "bot"))
        
        # Generate 10 "Wrong" Bot entries (Human-like behavior, but labeled 'bot') - UPDATED COUNT
        print(f"  Generating 10 'wrong' bot entries (human-like behavior, labeled bot)...")
        for _ in range(120): # Changed from 300 to 10
            all_entries.append(create_entry(product_name, human_params, "bot"))
        
        # Generate 10 "Wrong" Human entries (Bot-like behavior, but labeled 'human') - UPDATED COUNT
        print(f"  Generating 10 'wrong' human entries (bot-like behavior, labeled human)...")
        for _ in range(120): # Changed from 300 to 10
            all_entries.append(create_entry(product_name, bot_params, "human"))
            
        # Shuffle the entries to mix all types
        random.shuffle(all_entries)
        
        # Create DataFrame
        df = pd.DataFrame(all_entries)
        
        # Save to Excel file
        filename = f"training_{product_name.lower().replace(' ', '_')}_extended.xlsx"
        df.to_excel(filename, index=False)
        
        print(f"  ✅ Created {filename} with {len(all_entries)} entries")
        print(f"    - Correct Human entries (based on behavior): {len([e for e in all_entries if e['Label'] == 'human' and e['BuyingTime_Seconds'] >= 60])}")
        print(f"    - Correct Bot entries (based on behavior): {len([e for e in all_entries if e['Label'] == 'bot' and e['BuyingTime_Seconds'] <= 15])}")
        print(f"    - Wrong Bot (Human-like behavior, labeled bot) entries: {len([e for e in all_entries if e['Label'] == 'bot' and e['BuyingTime_Seconds'] >= 60])}")
        print(f"    - Wrong Human (Bot-like behavior, labeled human) entries: {len([e for e in all_entries if e['Label'] == 'human' and e['BuyingTime_Seconds'] <= 15])}")
        print()

def display_sample_data_extended():
    """
    Display sample data from the first extended training file for verification
    """
    try:
        # Read the first extended training file
        df = pd.read_excel("training_laptop_extended.xlsx")
        
        print("="*80)
        print("SAMPLE DATA FROM TRAINING_LAPTOP_EXTENDED.XLSX")
        print("="*80)
        
        # Show first 3 human entries
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
        
        # Show first 3 bot entries
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
        print("Extended training files not found. Please run generate_extended_training_data() first.")

if __name__ == "__main__":
    print("🚀 Starting Extended Training Data Generation...")
    print("="*60)
    
    # Generate extended training data
    generate_extended_training_data()
    
    print("="*60)
    print("✅ Extended training data generation completed!")
    print("\nNew files created:")
    print("- training_laptop_extended.xlsx")
    print("- training_jeans_extended.xlsx") 
    print("- training_decoration_lamp_extended.xlsx")
    print("\nEach file contains approximately:")
    print("- 10,000 correct human behavior entries")
    print("- 10,000 correct bot behavior entries")
    print("- 10 human-like entries labeled as bot (wrong bot)")
    print("- 10 bot-like entries labeled as human (wrong human)")
    print("- Total: 20,020 entries per product") # Updated total count
    print("="*60)
    
    # Display sample data from the extended files
    display_sample_data_extended()
