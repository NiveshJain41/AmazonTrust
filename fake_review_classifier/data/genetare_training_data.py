import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import uuid
import os

class MockDataGenerator:
    def __init__(self):
        # Product descriptions for generating hijacked reviews
        self.product_descriptions = {
            "Vivo Mobile Phone": "smartphone camera battery display performance android",
            "Cotton Shirt": "fabric cotton comfortable fit size quality material",
            "Amazon Prime Video": "streaming movies shows content quality subscription"
        }
        
        # Other product references for hijacked reviews
        self.other_products = {
            "Vivo Mobile Phone": ["Samsung Galaxy", "iPhone", "OnePlus", "Xiaomi"],
            "Cotton Shirt": ["Denim Jacket", "Polo Shirt", "T-shirt", "Hoodie"],
            "Amazon Prime Video": ["Netflix", "Disney+", "Hulu", "YouTube Premium"]
        }
        
        # Grammar and spelling errors for different review types
        self.common_errors = [
            ("good", "gud"), ("great", "grt"), ("very", "vry"), ("nice", "nyce"),
            ("quality", "qualaty"), ("product", "prodct"), ("amazing", "amzing"),
            ("excellent", "excelent"), ("perfect", "perfct"), ("recommend", "recomend")
        ]
        
        # Special characters for scripted reviews
        self.special_chars = ["!", "@", "#", "$", "%", "^", "&", "*", "~", "`"]
        
    def generate_human_reviews(self, product_name, count=50):
        """
        Generate realistic human reviews with varied timestamps, unique IPs,
        and natural language patterns.
        """
        reviews = []
        base_time = datetime.now()
        
        # Human reviews templates with natural language
        human_templates = [
            "I bought this {product} last month and I'm really happy with it. The quality is good and it works as expected.",
            "Great {product}! I would definitely recommend it to others. Worth the money.",
            "This {product} exceeded my expectations. The build quality is solid and it performs well.",
            "I've been using this {product} for a few weeks now and it's been great. No complaints so far.",
            "Pretty satisfied with this {product}. It does what it's supposed to do and the price is reasonable.",
            "Good value for money. The {product} works fine and I'm happy with my purchase.",
            "I was a bit skeptical at first but this {product} turned out to be really good. Glad I bought it.",
            "Decent {product}. Not the best I've used but definitely worth the price. Would buy again.",
            "This {product} is exactly what I needed. Works perfectly and the quality is impressive.",
            "Very pleased with this {product}. It arrived quickly and works as described in the listing."
        ]
        
        for i in range(count):
            template = random.choice(human_templates)
            review_text = template.format(product=product_name.lower())
            
            # Add some natural variations and length to human reviews
            if random.random() < 0.3:
                review_text += f" I've had it for {random.randint(1, 6)} months now."
            if random.random() < 0.2:
                review_text += " Highly recommend!"
            
            reviews.append({
                'product_name': product_name,
                'review': review_text,
                'label': 'Human',
                'stars': random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0], # Mostly positive
                'typing_time_seconds': random.uniform(180, 450), # Longer typing times
                'bought': 'Yes', # Humans usually buy the product
                'ip_address': self.generate_unique_ip(), # Unique IP for human reviews
                'account_id': f"A{random.randint(10**11, 10**12-1)}",
                'product_specific': 'Yes', # Reviews are specific to the product
                'helpful_votes_exists': random.choice(['Yes', 'Ask user']),
                'helpful_votes_percent': f"{random.randint(40, 85)}%",
                'device_fingerprint': 'Unique',
                'timestamp': (base_time - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d %H:%M:%S"),
                'review_id': str(uuid.uuid4())[:8]
            })
            
        return reviews
    
    def generate_script_reviews(self, product_name, count=50):
        """
        Generate scripted/bot reviews characterized by generic phrases,
        quick succession timestamps, shared IPs, and extreme ratings.
        """
        reviews = []
        base_time = datetime.now()
        script_ip = "192.168.10.100"  # Shared IP for script reviews
        
        # Generic scripted templates
        script_templates = [
            "Amazing product! Highly recommend it to everyone.",
            "Best value for money. Will buy again.",
            "Very satisfied. Works as described.",
            "Affordable and reliable. Go for it!",
            "Top-notch quality at this price.",
            "Excellent quality and fast delivery.",
            "Perfect for daily use. Great purchase!",
            "Outstanding performance. Worth every penny.",
            "Superb build quality. Exceeded expectations.",
            "Fantastic product. Will recommend to friends."
        ]
        
        # Generate reviews in batches with quick succession (script behavior)
        batch_time = base_time
        for i in range(count):
            review_text = random.choice(script_templates)
            
            # Add random special characters and spelling errors for scripted reviews
            if random.random() < 0.4:
                review_text += random.choice(self.special_chars) * random.randint(1, 3)
            
            # Add spelling errors occasionally to make them look less perfect
            if random.random() < 0.3:
                for correct, wrong in random.sample(self.common_errors, 1):
                    review_text = review_text.replace(correct, wrong)
            
            # Reviews posted in quick succession (script behavior simulation)
            if i % 10 == 0:  # New batch starting time every 10 reviews
                batch_time = base_time - timedelta(days=random.randint(1, 30))
            
            reviews.append({
                'product_name': product_name,
                'review': review_text,
                'label': 'Script',
                'stars': random.choice([1, 5]),  # Extreme ratings common for scripts
                'typing_time_seconds': random.uniform(3, 15), # Very short typing times
                'bought': 'No', # Scripts often don't simulate purchase
                'ip_address': script_ip, # Shared IP indicates bot activity
                'account_id': f"A{random.randint(10**11, 10**12-1)}",
                'product_specific': 'No', # Generic reviews
                'helpful_votes_exists': 'No',
                'helpful_votes_percent': '0%',
                'device_fingerprint': 'Reused',
                'timestamp': (batch_time + timedelta(seconds=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S"),
                'review_id': str(uuid.uuid4())[:8]
            })
            
        return reviews
    
    def generate_ai_reviews(self, product_name, count=50):
        """
        Generate AI-generated reviews characterized by more sophisticated language,
        product-specific keywords, and often instant generation.
        """
        reviews = []
        base_time = datetime.now()
        ai_ip = "10.0.0.50"  # Shared IP for AI reviews
        
        # AI templates - more product-specific but still generic-sounding, sometimes verbose
        ai_templates = [
            "This {product} delivers exceptional performance with its advanced features. The user experience is seamless and intuitive.",
            "I'm impressed by the quality and functionality of this {product}. It meets all my requirements perfectly.",
            "The {product} offers excellent value proposition with its comprehensive feature set and reliable performance.",
            "Outstanding {product} that combines innovation with practicality. Highly satisfied with the overall experience.",
            "This {product} stands out in its category with superior build quality and impressive specifications.",
            "Remarkable {product} that exceeds industry standards. The attention to detail is commendable.",
            "The {product} provides optimal performance and user satisfaction. A worthwhile investment indeed.",
            "Exceptional {product} with cutting-edge technology and user-friendly design. Thoroughly recommended.",
            "This {product} offers unparalleled quality and performance in its price range. Excellent choice.",
            "Premium {product} with outstanding features and reliable functionality. Completely satisfied with purchase."
        ]
        
        # Generate reviews posted simultaneously or in quick batches (AI generation behavior)
        # Adjust batch_times calculation to avoid index errors if count is small
        num_batches = max(1, count // 10) # Ensure at least one batch
        batch_times = [base_time - timedelta(days=random.randint(1, 60)) for _ in range(num_batches)]
        
        for i in range(count):
            template = random.choice(ai_templates)
            review_text = template.format(product=product_name.lower())
            
            # Add product-specific keywords to make AI reviews seem relevant
            if product_name in self.product_descriptions and random.random() < 0.5:
                keywords = self.product_descriptions[product_name].split()
                if keywords: # Ensure keywords list is not empty
                    review_text += f" The {random.choice(keywords)} is particularly impressive."
            
            # Same timestamp for a batch of reviews (AI generation behavior simulation)
            batch_index = i // 10
            # Ensure batch_index does not exceed the bounds of batch_times
            timestamp = batch_times[min(batch_index, len(batch_times) - 1)]
            
            reviews.append({
                'product_name': product_name,
                'review': review_text,
                'label': 'AI',
                'stars': random.choice([1, 5]),  # Extreme ratings for AI too
                'typing_time_seconds': 0,  # Instant generation for AI
                'bought': 'No', # AI often doesn't simulate purchase
                'ip_address': ai_ip, # Shared IP
                'account_id': f"A{random.randint(10**11, 10**12-1)}",
                'product_specific': 'Yes',  # AI can be product-specific in text
                'helpful_votes_exists': 'No',
                'helpful_votes_percent': f"{random.randint(5, 15)}%",
                'device_fingerprint': 'Reused',
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'review_id': str(uuid.uuid4())[:8]
            })
            
        return reviews
    
    def generate_hijacked_reviews(self, product_name, count=50):
        """
        Generate hijacked reviews which are legitimate reviews for a *different* product
        but are repurposed for the target product. Characterized by irrelevant content
        and older timestamps.
        """
        reviews = []
        base_time = datetime.now()
        
        # Hijacked reviews reference other products, making them irrelevant to the current product
        # Ensure product_name exists in other_products before accessing
        if product_name in self.other_products:
            other_product_reference = random.choice(self.other_products[product_name])
        else:
            # Fallback if product_name not in other_products, to prevent errors
            other_product_reference = "another unrelated product" 
        
        hijacked_templates = [
            "I bought this {other_product} and it's been amazing. Great quality and fast shipping.",
            "This {other_product} is exactly what I needed. Works perfectly and arrived on time.",
            "Excellent {other_product}! Very happy with the quality and performance. Highly recommend.",
            "Great value {other_product}. Been using it for months without any issues.",
            "Perfect {other_product} for the price. Quality is impressive and delivery was quick.",
            "This {other_product} exceeded my expectations. Well built and reliable.",
            "Outstanding {other_product}! Great features and excellent customer service.",
            "Very satisfied with this {other_product}. Good quality and reasonable price.",
            "Fantastic {other_product}! Works as described and shipping was fast.",
            "This {other_product} is a great purchase. High quality and good value for money."
        ]
        
        for i in range(count):
            template = random.choice(hijacked_templates)
            review_text = template.format(other_product=other_product_reference)
            
            # Hijacked reviews are typically old (e.g., exactly 1 year old)
            old_date = base_time - timedelta(days=365)
            
            reviews.append({
                'product_name': product_name,
                'review': review_text,
                'label': 'Hijacked',
                'stars': random.randint(4, 5),  # High ratings to seem legitimate
                'typing_time_seconds': random.uniform(200, 400),  # Normal typing time (as if a real user wrote it)
                'bought': 'Yes',  # Appears legitimate
                'ip_address': self.generate_unique_ip(), # Unique IP (as it was a real user)
                'account_id': f"A{random.randint(10**11, 10**12-1)}",
                'product_specific': 'Misleading',  # Crucially, references another product
                'helpful_votes_exists': 'Yes',
                'helpful_votes_percent': f"{random.randint(20, 60)}%",
                'device_fingerprint': 'Unique',
                'timestamp': old_date.strftime("%Y-%m-%d %H:%M:%S"),
                'review_id': str(uuid.uuid4())[:8]
            })
            
        return reviews
    
    def generate_unique_ip(self):
        """Generate a random, unique IP address."""
        return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    def generate_incorrect_entries(self, num_entries=15, product_name="Vivo Mobile Phone"):
        """
        Generates reviews that have the characteristics of one label (e.g., Human)
        but are intentionally assigned a different, incorrect label (e.g., AI).
        This is designed to introduce noise and prevent a machine learning model
        from achieving 100% accuracy on the dataset.
        """
        incorrect_reviews = []
        labels = ['Human', 'AI', 'Script', 'Hijacked']
        
        # Determine how many incorrect entries each 'true' type will contribute
        entries_per_true_type = num_entries // len(labels)
        remainder = num_entries % len(labels)

        # Map labels to their respective generator functions
        generator_map = {
            'Human': self.generate_human_reviews,
            'AI': self.generate_ai_reviews,
            'Script': self.generate_script_reviews,
            'Hijacked': self.generate_hijacked_reviews
        }

        # Generate reviews with their 'true' characteristics, then mislabel them
        for i, original_label in enumerate(labels):
            # Distribute the `num_entries` as evenly as possible
            count_for_this_type = entries_per_true_type + (1 if i < remainder else 0)
            if count_for_this_type == 0: # Skip if no entries needed for this type
                continue
                
            # Generate reviews using the *correct* generator for the content type's characteristics
            true_reviews = generator_map[original_label](product_name, count=count_for_this_type)
            
            for review in true_reviews:
                # Randomly assign an *incorrect* label (must be different from original_label)
                possible_incorrect_labels = [lbl for lbl in labels if lbl != original_label]
                if possible_incorrect_labels: # Ensure there are labels to choose from
                    review['label'] = random.choice(possible_incorrect_labels)
                else:
                    # Fallback in case of an unlikely scenario (e.g., only one label defined)
                    review['label'] = original_label # Keep original if no other options
                incorrect_reviews.append(review)
        
        return incorrect_reviews

    def create_training_dataset(self):
        """
        Create the complete training dataset, including 500 correct entries
        for each type and 15 intentionally incorrect (mislabeled) entries.
        """
        all_reviews = []
        # Using a single product name for all generated reviews as specified
        product_name = "Vivo Mobile Phone" 
        
        print(f"Generating 500 correct reviews for each type for '{product_name}'...")
        
        # Generate 500 correctly labeled samples of each type
        human_reviews = self.generate_human_reviews(product_name, 500)
        script_reviews = self.generate_script_reviews(product_name, 500)
        ai_reviews = self.generate_ai_reviews(product_name, 500)
        hijacked_reviews = self.generate_hijacked_reviews(product_name, 500)
        
        all_reviews.extend(human_reviews)
        all_reviews.extend(script_reviews)
        all_reviews.extend(ai_reviews)
        all_reviews.extend(hijacked_reviews)

        print(f"Generating 15 incorrect (mislabeled) entries...")
        # Generate the 15 incorrect entries
        incorrect_entries = self.generate_incorrect_entries(num_entries=15, product_name=product_name)
        all_reviews.extend(incorrect_entries)
        
        # Create DataFrame from all generated reviews
        df = pd.DataFrame(all_reviews)
        
        # Shuffle the entire dataset to mix correct and incorrect entries
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Generated {len(df)} total training samples.")
        print("\nDistribution of labels (after adding incorrect entries):")
        print(df['label'].value_counts())
        print("\nProduct distribution (should be mostly one product):")
        print(df['product_name'].value_counts())
        
        return df
    
    def save_training_data(self, output_file="data/training_data.xlsx"):
        """
        Generates the training data and saves it to an Excel file.
        Creates the 'data' directory if it doesn't exist.
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate training dataset
        df = self.create_training_dataset()
        
        # Save the DataFrame to an Excel file
        df.to_excel(output_file, index=False)
        print(f"\nTraining data saved to: {output_file}")
        
        return df

# Usage example to generate and save the data
if __name__ == "__main__":
    generator = MockDataGenerator()
    training_df = generator.save_training_data()
    
    # Display sample data from the generated DataFrame
    print("\nSample data (first 5 rows):")
    print(training_df.head())
    
    print("\nSample reviews by type (showing actual label from dataframe):")
    # Loop through each label to show an example review for each category
    for label in ['Human', 'Script', 'AI', 'Hijacked']:
        # Ensure there's at least one entry for the current label before trying to access iloc[0]
        if not training_df[training_df['label'] == label].empty:
            print(f"\n--- {label} Review Example ---")
            sample = training_df[training_df['label'] == label].iloc[0]
            print(f"Review: {sample['review']}")
            print(f"Assigned Label: {sample['label']}") # Display the actual label in the dataframe
            print(f"Stars: {sample['stars']}, Typing time: {sample['typing_time_seconds']:.2f}s")
            print(f"IP Address: {sample['ip_address']}, Bought: {sample['bought']}")
            print(f"Product Specific: {sample['product_specific']}, Helpful Votes: {sample['helpful_votes_percent']}")
            print(f"Timestamp: {sample['timestamp']}")
            print(f"Review ID: {sample['review_id']}")
        else:
            print(f"\nNo reviews found for label: {label} in the generated dataset (this should not happen with current counts).")

