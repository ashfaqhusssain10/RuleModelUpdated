import pandas as pd
import numpy as np
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBRegressor
# Suppress FutureWarnings globally
warnings.simplefilter(action='ignore', category=FutureWarning)



class FoodItemMatcher:
    """Two-tier hash table system for efficient food item matching."""

    def __init__(self, item_metadata=None):
        self.direct_lookup = {}  # Tier 1: Direct lookup hash table
        self.token_to_items = {}  # Tier 2: Token-based lookup
        self.category_items = {}  # Items organized by category

        if item_metadata:
            self.build_hash_tables(item_metadata)

    def build_hash_tables(self, item_metadata):
        """Build the two-tier hash table system from item metadata."""
        # Reset tables
        self.direct_lookup = {}
        self.token_to_items = {}
        self.category_items = {}

        # Process each item
        for item_name, metadata in item_metadata.items():
            if not item_name or pd.isna(item_name):
                continue
            if isinstance(item_name,tuple) and len(item_name) > 0:
                item_name = item_name[0]

            # Standardize name
            std_name = self._standardize_name(item_name)
            if 'sambar' in std_name.lower():
                print(f"DEBUG: Adding '{std_name}' to direct lookup from '{item_name}'")
            # Add to direct lookup (Tier 1)
            self.direct_lookup[std_name] = item_name

            # Add common variants
            variants = self._generate_variants(std_name)
            for variant in variants:
                if variant not in self.direct_lookup:
                    self.direct_lookup[variant] = item_name

            # Add to token lookup (Tier 2)
            tokens = self._tokenize(std_name)
            for token in tokens:
                if token not in self.token_to_items:
                    self.token_to_items[token] = []
                if item_name not in self.token_to_items[token]:
                    self.token_to_items[token].append(item_name)

            # Add to category items
            category = metadata.category
            if category not in self.category_items:
                self.category_items[category] = []
            self.category_items[category].append(item_name)

    def find_item(self, query_item, item_metadata):
        """Find a food item using the two-tier hash table approach."""
        if not query_item or pd.isna(query_item):
            return None, None

        # Standardize query - handle '>' prefix automatically
        std_query = self._standardize_name(query_item)
        std_query = std_query.replace('> ', '').replace('>', '')  # Remove '>' prefixes
        print(f"DEBUG: Looking for '{std_query}'")
        print(f"DEBUG: Item metadata keys: {[self._standardize_name(k) for k in item_metadata.keys() if 'sambar' in self._standardize_name(k).lower()]}")
        exact_match = any(self._standardize_name(k) == std_query for k in item_metadata.keys())
        print(f"DEBUG: Exact match exists: {exact_match}")
        for item_name in item_metadata:
            if self._standardize_name(item_name)==std_query:
                # If there's an exact match in metadata, prioritize that immediately
                return item_name,item_metadata[item_name]

        # Tier 1: Try direct lookup first (fast path)
        if std_query in self.direct_lookup:
            item_name = self.direct_lookup[std_query]
            # Ensure item_name is a string, not a tuple
            if isinstance(item_name, tuple) and len(item_name) > 0:
                item_name = item_name[0]
            if item_name in item_metadata:
                return item_name, item_metadata[item_name]

        # Additional direct lookups for common variations
        # Try with spaces removed
        compact_query = std_query.replace(' ', '')
        for key in self.direct_lookup:
            compact_key = key.replace(' ', '')
            if compact_query == compact_key:
                item_name = self.direct_lookup[key]
                # Ensure item_name is a string, not a tuple
                if isinstance(item_name, tuple) and len(item_name) > 0:
                    item_name = item_name[0]
                if item_name in item_metadata:
                    return item_name, item_metadata[item_name]

        # Tier 2: Enhanced token-based lookup
        tokens = self._tokenize(std_query)
        if tokens:
            # Find candidates with improved scoring
            candidates = {}
            for token in tokens:
                # Handle token variations (plurals, singulars)
                token_variants = [token]
                if token.endswith('s'):
                    token_variants.append(token[:-1])  # Remove 's' for plurals
                elif len(token) > 3:
                    token_variants.append(token + 's')  # Add 's' for singulars

                for variant in token_variants:
                    if variant in self.token_to_items:
                        for item_name in self.token_to_items[variant]:
                            # Ensure item_name is a string, not a tuple
                            if isinstance(item_name, tuple) and len(item_name) > 0:
                                item_name = item_name[0]
                            if item_name not in candidates:
                                candidates[item_name] = 0
                            candidates[item_name] += 1

            # Enhance scoring with additional contextual factors
            scored_candidates = []
            for item_name, token_matches in candidates.items():
                if item_name in item_metadata:
                    item_tokens = self._tokenize(self._standardize_name(item_name))
                    if not item_tokens:
                        continue

                    # Basic token match score
                    token_score = token_matches / max(len(tokens), len(item_tokens))

                    # Enhanced substring matching
                    contains_score = 0
                    if std_query in self._standardize_name(item_name):
                        contains_score = 0.8
                    elif self._standardize_name(item_name) in std_query:
                        contains_score = 0.6

                    # Word overlap score (considering word position)
                    word_overlap = 0
                    std_query_words = std_query.split()
                    item_words = self._standardize_name(item_name).split()
                    for i, qword in enumerate(std_query_words):
                        for j, iword in enumerate(item_words):
                            if qword == iword:
                                # Words in same position get higher score
                                pos_factor = 1.0 - 0.1 * abs(i - j)
                                word_overlap += pos_factor

                    if len(std_query_words) > 0:
                        word_overlap_score = word_overlap / len(std_query_words)
                    else:
                        word_overlap_score = 0

                    # Combined score with weights
                    final_score = max(token_score * 0.4 + contains_score * 0.4 + word_overlap_score * 0.2,
                                      contains_score)

                    scored_candidates.append((item_name, final_score))

            # Sort by score and get best match
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_match = scored_candidates[0]

                # Lower threshold for matching (0.4 instead of 0.5)
                if best_match[1] >= 0.4:
                    item_name = best_match[0]
                    # Final check to ensure item_name is a string
                    if isinstance(item_name, tuple) and len(item_name) > 0:
                        item_name = item_name[0]
                    return item_name, item_metadata[item_name]

        return None, None

    def _standardize_name(self, name):
        """Standardize item name for matching."""
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = " ".join(name.split())  # Normalize whitespace
        return name

    def _tokenize(self, text):
        """Split text into tokens for matching."""
        if not text:
            return []

        # Simple tokenization by whitespace
        tokens = text.split()

        # Remove very common words and short tokens
        stop_words = {"and", "with", "the", "in", "of", "a", "an"}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        return tokens

    def _generate_variants(self, name):
        """Generate common variants of a food item name."""
        variants = []

        # Common misspellings and variations
        replacements = {
            "biryani": ["biriyani", "briyani", "biryani"],
            "chicken": ["chiken", "chikken", "checken"],
            "paneer": ["panner", "panir", "pnr"],
            "masala": ["masala", "masaala", "masalla"]
        }

        # Generate simple word order variations
        words = name.split()
        if len(words) > 1:
            # Add reversed order for two-word items
            variants.append(" ".join(reversed(words)))

        # Apply common spelling variations
        for word, alternatives in replacements.items():
            if word in name:
                for alt in alternatives:
                    variant = name.replace(word, alt)
                    variants.append(variant)

        return variants


# Configure logging for FoodCategoryRules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("food_rules.log"), logging.StreamHandler()]
)
logger = logging.getLogger("FoodCategoryRules")

class FoodCategoryRules:
    def __init__(self):
        logger.info("Initializing FoodCategoryRules")
        self.category_rules = self._initialize_category_rules()
        self.category_dependencies = self._initialize_category_dependencies()
        self.meal_type_modifiers = self._initialize_meal_type_modifiers()

    def _initialize_category_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Welcome_Drinks": {
                "min_quantity": "100ml",
                "max_quantity": "120ml",
                "default_quantity": "100ml",
                "vc_price": 180,
                "adjustments": {
                    "large_event": lambda guest_count: "120ml" if guest_count > 50 else "120ml"
                }
            },
            "Biryani": {
                "min_quantity": "250g",
                "max_quantity": "450g",
                "default_quantity": "300g",
                "vc_price": 300,
                "adjustments": {
                    "per_person": lambda guest_count: "250g" if guest_count > 100 else "300g",
                    "multiple_varieties": lambda count: "250g" if count > 2 else "300g",
                    "total_items": lambda total_items: "450g" if total_items <= 3 else ("320g" if total_items <= 3 else "300g")
                }
            },
            "Salad": {
                "min_quantity": "50g",
                "max_quantity": "50g",
                "default_quantity": "50g",
                "vc_price": 70,
            },
            "Podi": {
                "min_quantity": "10g",
                "max_quantity": "10g",
                "default_quantity": "10g",
                "vc_price": 100,
            },
            "Fried_Items": {
                "min_quantity": "40g",
                "max_quantity": "60g",
                "default_quantity": "50g",
                "vc_price": 100,
            },
            "Ghee": {
                "min_quantity": "3g",
                "max_quantity": "5g",
                "default_quantity": "3g",
                "vc_price": 150,
            },
            "Pickle": {
                "min_quantity": "10g",
                "max_quantity": "10g",
                "default_quantity": "10g",
                "vc_price": 250,
            },
            "Flavored_Rice": {
                "min_quantity": "80g",
                "max_quantity": "100g",
                "default_quantity": "100g",
                "vc_price": 150,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "100g",
                        2: "80g",
                        3: "80g",
                        4: "80g",
                        5: "80g",
                    }.get(count, "80g")
                }
            },
            "Soups": {
                "min_quantity": "120ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 150,
            },
            "Crispers": {
                "min_quantity": "5g",
                "max_quantity": "5g",
                "default_quantity": "5g",
                "vc_price": 40,
            },
            "Fried_Rice": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "80g",
                "vc_price": 70,
                "adjustments": {
                    "per_person": lambda guest_count: "100g" if guest_count > 100 else "100g",
                    "multiple_varieties": lambda count: "80g" if count > 3 else "100g"
                }
            },
            "Fry": {
                "min_quantity": "40g",
                "max_quantity": "60g",
                "default_quantity": "40g",
                "vc_price": 60,
            },
            "Fryums": {
                "min_quantity": "5g",
                "max_quantity": "10g",
                "default_quantity": "5g",
                "vc_price": 10,
            },
            "Salan": {
                "min_quantity": "40g",
                "max_quantity": "40g",
                "default_quantity": "40g",
                "vc_price": 80,
            },
            "Cakes": {
                "min_quantity": "500g",
                "max_quantity": "1000g",
                "default_quantity": "500g",
                "vc_price": 400,
                "adjustments": {
                    "per_person": lambda guest_count: "500g" if guest_count > 30 else "1000g"
                }
            },
            "Cup_Cakes": {
                "min_quantity": "50g",
                "max_quantity": "100g",
                "default_quantity": "50g",
                "vc_price": 200,
            },
            "Hot_Beverages": {
                "min_quantity": "100ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 200,
            },
            "Pulav": {
                "min_quantity": "250g",
                "max_quantity": "450g",
                "default_quantity": "300g",
                "vc_price": 300,
                "adjustments": {
                    "per_person": lambda guest_count: "250g" if guest_count > 100 else "300g",
                    "multiple_varieties": lambda count: "250g" if count > 2 else "300g"
                }
            },
            "Appetizers": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 270,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "120g",
                        2: "100g",
                        3: "80g",
                        4: "60g",
                        5: "50g",
                        6: "50g",
                        7: "50g",
                        8: "50g",
                        9: "50g",
                        10: "50g"
                    }.get(count, "80g")
                }
            },
            "Roti_Pachadi": {
                "min_quantity": "10g",
                "max_quantity": "20g",
                "default_quantity": "10g",
                "vc_price": 80,
            },
            "Curries": {
                "min_quantity": "120g",
                "max_quantity": "180g",
                "default_quantity": "120g",
                "vc_price": 270,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "120g",
                        2: "100g",
                        3: "80g",
                        4: "80g",
                        5: "80g",
                        6: "80g",
                    }.get(count, "100g")
                }
            },
            "Rice": {
                "min_quantity": "150g",
                "max_quantity": "250g",
                "default_quantity": "200g",
                "vc_price": 80,
                "adjustments": {
                    "with_curry": lambda: "200g",
                    "standalone": lambda: "250g"
                }
            },
            "Liquids(Less_Dense)": {
                "min_quantity": "60ml",
                "max_quantity": "100ml",
                "default_quantity": "70ml",
                "vc_price": 100,
                "adjustments": {
                    "with_dal": lambda: "60ml",
                    "standalone": lambda: "100ml"
                }
            },
            "Liquids(High_Dense)": {
                "min_quantity": "30ml",
                "max_quantity": "30ml",
                "default_quantity": "30ml",
                "vc_price": 160,
            },
            "Dal": {
                "min_quantity": "60g",
                "max_quantity": "80g",
                "default_quantity": "70g",
                "vc_price": 120,
            },
            "Desserts": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 170,
                "adjustments": {
                    "variety_count": lambda count: "80g" if count > 2 else "100g"
                }
            },
            "Curd": {
                "min_quantity": "50g",
                "max_quantity": "60g",
                "default_quantity": "50g",
                "vc_price": 50,
            },
            "Fruits": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 150,
            },
            "Paan": {
                "min_quantity": "1pcs",
                "max_quantity": "1pcs",
                "default_quantity": "1pcs",
                "vc_price": 50,
            },
            "Omlette": {
                "min_quantity": "1pcs",
                "max_quantity": "1pcs",
                "default_quantity": "1pcs",
                "vc_price": 150,
            },
            "Breads": {
                "min_quantity": "1pcs",
                "max_quantity": "2pcs",
                "default_quantity": "1pcs",
                "vc_price": 40,
                "adjustments": {
                    "with_curry": lambda: "1pcs"
                }
            },
            "Italian": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 300,
                "adjustments": {
                    "with_pasta": lambda: "120g"
                }
            },
            "Pizza": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 200,
            },
            "Chutney":{
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 40,
            },
            "Raitha": {
                "min_quantity": "50g",
                "max_quantity": "60g",
                "default_quantity": "60g",
                "vc_price": 100,
                "adjustments": {
                    "with_biryani": lambda: "70g",
                    "standalone": lambda: "50g"
                }
            },
            "Dips":{
                "min_quantity": "10g",
                "max_quantity": "20g",
                "default_quantity": "15",
                "vc_price": 70,
            },
            "Breakfast": {
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 150,
            },
            "Sandwich": {
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 150,
            }
        }

    def _initialize_category_dependencies(self) -> Dict[str, List[str]]:
        return {
            "Pulav": ["Raita", "Salad"],
            "Rice": ["Curries", "Liquids(Less_Dense)", "Curd"],
            "Breads": ["Curries"],
            "Curries": ["Rice", "Breads"],
            "Dal": ["Rice", "Breads"],
            "Salan": ["Biryani", "Pulav"],
            "Curd": ["Rice"],
            "Rasam": ["Rice", "Dal"],
            "Chutneys": ["Rice"],
            "Roti_Pachadi": ["Rice"],
            "Pickles": ["Rice", "Flavoured_Rice"],
            "Podi": ["Rice", "Flavoured_Rice"]
        }

    def _initialize_meal_type_modifiers(self) -> Dict[str, float]:
        return {
            "Breakfast": 0.8,
            "Lunch": 1.0,
            "Hi-Tea": 0.9,
            "Dinner": 1.0
        }

    def extract_quantity_value(self, quantity_str: str) -> float:
        if pd.isna(quantity_str):
            return 0.0
        match = re.search(r'(\d+\.?\d*)', str(quantity_str))
        if match:
            return float(match.group(1))
        return 0.0

    def extract_unit(self, quantity_str: str) -> str:
        if pd.isna(quantity_str):
            return ''
        match = re.search(r'[a-zA-Z]+', str(quantity_str))
        if match:
            return match.group(0)
        return ''

    def infer_default_unit(self, category: str) -> str:
        liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
        piece_categories = ["Breads", "Paan", "Omlette"]
        if category in liquid_categories:
            return "ml"
        elif category in piece_categories:
            return "pcs"
        else:
            return "g"

    def validate_unit(self, category: str, unit: str) -> str:
        liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
        piece_categories = ["Breads", "Paan", "Omlette"]
        if not unit:
            inferred_unit = self.infer_default_unit(category)
            return inferred_unit
        if category in liquid_categories and unit not in ["ml", "l"]:
            logger.warning(f"Invalid unit '{unit}' for liquid category '{category}', defaulting to 'ml'")
            return "ml"
        elif category in piece_categories and unit != "pcs":
            logger.warning(f"Invalid unit '{unit}' for piece category '{category}', defaulting to 'pcs'")
            return "pcs"
        elif unit not in ["g", "kg", "ml", "l", "pcs"]:
            logger.warning(f"Invalid unit '{unit}' for category '{category}', defaulting to 'g'")
            return "g"
        return unit

    def normalize_category_name(self, category: str) -> str:
        return category.replace(" ", "_").strip()

    def get_default_quantity(self, category: str) -> Tuple[float, str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category in self.category_rules:
            rule = self.category_rules[normalized_category]
            default_qty_str = rule["default_quantity"]
            default_qty = self.extract_quantity_value(default_qty_str)
            unit = self.extract_unit(default_qty_str)
            unit = self.validate_unit(normalized_category, unit)
            return default_qty, unit
        inferred_unit = self.infer_default_unit(normalized_category)
        return 0.0, inferred_unit

    def get_item_specific_quantity(self, item_name, guest_count, item_specific_data):
        """
        Get item-specific quantity based on per-guest ratio.

        Args:
            item_name: Name of the item (string or tuple)
            guest_count: Number of guests
            item_specific_data: Dictionary of item-specific data

        Returns:
            Tuple of (quantity, unit) or (None, None) if not available
        """
        # Handle if item_name is a tuple (output from FoodItemMatcher)
        if isinstance(item_name, tuple) and len(item_name) > 0:
            item_name = item_name[0]  # Extract the string from the tuple

        # Now proceed with the standard processing
        std_item_name = item_name.lower().strip() if isinstance(item_name, str) else ""
        if std_item_name not in item_specific_data:
            return None, None

        item_data = item_specific_data[std_item_name]
        preferred_unit = item_data['preferred_unit']
        per_guest_ratio = item_data['per_guest_ratio']

        if preferred_unit == "pcs" and per_guest_ratio is not None:
            quantity = guest_count * per_guest_ratio
            return quantity, "pcs"
        return None, None

    def apply_category_rules(self, category: str, guest_count: int, item_count: int, **kwargs) -> Tuple[float, str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category not in self.category_rules:
            qty, unit = self.get_default_quantity(normalized_category)
            return qty, unit
        rule = self.category_rules[normalized_category]
        default_qty_str = rule["default_quantity"]
        default_qty = self.extract_quantity_value(default_qty_str)
        unit = self.extract_unit(default_qty_str)
        unit = self.validate_unit(normalized_category, unit)
        adjusted_qty = default_qty

        if "adjustments" in rule:
            if "large_event" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["large_event"](guest_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Large event adjustment = {adjusted_qty_str}")
            if "per_person" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["per_person"](guest_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Per-person adjustment = {adjusted_qty_str}")
            if "multiple_varieties" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["multiple_varieties"](item_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Multiple varieties adjustment ({item_count}) = {adjusted_qty_str}")
            if "variety_count" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["variety_count"](item_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Variety count adjustment ({item_count}) = {adjusted_qty_str}")
            if "total_items" in rule["adjustments"] and "total_items" in kwargs:
                total_items = kwargs.get("total_items")
                adjusted_qty_str = rule["adjustments"]["total_items"](total_items)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Total items adjustment ({total_items}) = {adjusted_qty_str}")
        return adjusted_qty, unit

    def apply_dependency_rules(self, category: str, dependent_category: str, current_qty: float, unit: str) -> float:
        normalized_category = self.normalize_category_name(category)
        if normalized_category not in self.category_rules:
            return current_qty
        rule = self.category_rules[normalized_category]
        if "adjustments" not in rule:
            return current_qty
        adjustments = rule["adjustments"]
        dependent_normalized = dependent_category.lower()

        if dependent_normalized == "breads" and "with_breads" in adjustments:
            adjusted_qty_str = adjustments["with_breads"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "rice" and "with_rice" in adjustments:
            adjusted_qty_str = adjustments["with_rice"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "curries" and "with_curry" in adjustments:
            adjusted_qty_str = adjustments["with_curry"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "biryani" and "with_biryani" in adjustments:
            adjusted_qty_str = adjustments["with_biryani"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "dal" and "with_dal" in adjustments:
            adjusted_qty_str = adjustments["with_dal"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "rasam" and "with_rasam" in adjustments:
            adjusted_qty_str = adjustments["with_rasam"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty

        return current_qty

    def apply_meal_type_modifier(self, meal_type: str, qty: float) -> float:
        if meal_type in self.meal_type_modifiers:
            modifier = self.meal_type_modifiers[meal_type]
            return qty * modifier
        return qty

    def get_dependent_categories(self, category: str) -> List[str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category in self.category_dependencies:
            return self.category_dependencies[normalized_category]
        return []
class SuppressSpecificWarnings(logging.Filter):
    def filter(self, record):
        # Suppress specific warning messages
        msg = record.getMessage()
        if "No match found for item" in msg or "Using default category 'Curries' for" in msg:
            return False  # Do not log these messages
        return True  # Log all other messages
# Configure logging for HierarchicalFoodPredictor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("food_predictor.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HierarchicalFoodPredictor")
logger.addFilter(SuppressSpecificWarnings())  # Apply the custom filter

@dataclass
class ItemMetadata:
    category: str
    unit: str
    conversion_factor: float = 1.0

class HierarchicalFoodPredictor:
    def __init__(self, category_constraints=None, item_vc_file=None, item_data_file="item_data.csv"):
        logger.info("Initializing HierarchicalFoodPredictor")
        self.category_models = {}
        self.item_models = {}
        self.category_scalers = {}
        self.item_scalers = {}
        self.event_time_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.meal_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.event_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.food_rules = FoodCategoryRules()
        self.custom_category_constraints = category_constraints or {}
        self.item_metadata = {}
        self.item_vc_mapping = {}
        self.item_matcher =  None
        self.item_specific_data = {}
        if item_vc_file:
            self.load_item_vc_data(item_vc_file)
        if item_data_file:
            self.load_item_specific_data(item_data_file)

    def load_item_specific_data(self, item_data_file):
        logger.info(f"Loading item-specific data from {item_data_file}")
        try:
            item_data = pd.read_csv(item_data_file)
            required_columns = ['item_name', 'category', 'preferred_unit', 'per_guest_ratio', 'base_price_per_piece', 'base_price_per_kg']
            missing_columns = [col for col in required_columns if col not in item_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {item_data_file}: {missing_columns}")

            for _, row in item_data.iterrows():
                item_name = self.standardize_item_name(row['item_name'])
                self.item_specific_data[item_name] = {
                    'category': row['category'],
                    'preferred_unit': row['preferred_unit'],
                    'per_guest_ratio': float(row['per_guest_ratio']) if pd.notna(row['per_guest_ratio']) else None,
                    'base_price_per_piece': float(row['base_price_per_piece']) if pd.notna(row['base_price_per_piece']) else None,
                    'base_price_per_kg': float(row['base_price_per_kg']) if pd.notna(row['base_price_per_kg']) else None
                }
            logger.info(f"Loaded item-specific data for {len(self.item_specific_data)} items")
        except Exception as e:
            logger.error(f"Failed to load item-specific data: {e}")
            raise

    def load_item_vc_data(self, item_vc_file):
        logger.info(f"Loading item VC data from {item_vc_file}")
        try:
            vc_data = pd.read_excel(item_vc_file)
            for _, row in vc_data.iterrows():
                item_name = self.standardize_item_name(row['Item_name'])
                self.item_vc_mapping[item_name] = {
                    'VC': float(row['VC']),
                    'p_value': float(row.get('Power Factor (p)', 0.18))
                }
            logger.info(f"Loaded VC and P_value data for {len(self.item_vc_mapping)} items")
        except Exception as e:
            logger.error(f"Failed to load item VC data: {e}")
            raise

    def extract_quantity_value(self, quantity_str):
        return self.food_rules.extract_quantity_value(quantity_str)

    def extract_unit(self, quantity_str):
        return self.food_rules.extract_unit(quantity_str)

    def prepare_features(self, data):
        logger.info("Preparing features")
        if not hasattr(self, 'encoders_fitted') or not self.encoders_fitted:
            self.event_time_encoder.fit(data[['Event_Time']])
            self.meal_type_encoder.fit(data[['Meal_Time']])
            self.event_type_encoder.fit(data[['Event_Type']])
            self.encoders_fitted = True

        event_time_encoded = self.event_time_encoder.transform(data[['Event_Time']])
        meal_type_encoded = self.meal_type_encoder.transform(data[['Meal_Time']])
        event_type_encoded = self.event_type_encoder.transform(data[['Event_Type']])
        features = np.hstack([event_time_encoded, meal_type_encoded, event_type_encoded])
        return features

    def standardize_item_name(self, item_name):
        if pd.isna(item_name):
            return ""
        item_name = str(item_name).strip()
        standardized = item_name.lower()
        standardized = " ".join(standardized.split())
        return standardized

    def find_closest_item(self, item_name):
        """Find the closest matching item using the FoodItemMatcher if available"""
        # First try FoodItemMatcher if available
        if hasattr(self, 'item_matcher') and self.item_matcher:
            matched_item, _ = self.item_matcher.find_item(item_name, self.item_metadata)
            if matched_item:
                logger.info(f"FoodItemMatcher matched '{item_name}' to '{matched_item}'")
                return matched_item

        # Fall back to existing logic if FoodItemMatcher fails or isn't available
        if not self.item_metadata:
            return None
        std_name = self.standardize_item_name(item_name)
        if std_name in self.item_name_mapping:
            return self.item_name_mapping[std_name]
        best_match = None
        best_score = 0
        for known_name in self.item_metadata:
            std_known = self.standardize_item_name(known_name)
            if std_known == std_name:
                logger.info(f"Exact match found for '{item_name}' to '{known_name}'")
                return known_name
            if std_name in std_known or std_known in std_name:
                similarity = min(len(std_name), len(std_known)) / max(len(std_name), len(std_known))
                if similarity > 0.99 and similarity > best_score:
                    best_score = similarity
                    best_match = known_name
                continue

            std_name_words = set(std_name.split())
            std_known_words = set(std_known.split())
            common_words = std_name_words.intersection(std_known_words)
            if common_words:
                similarity = len(common_words) / max(len(std_name_words), len(std_known_words))
                if similarity > best_score:
                    best_score = similarity
                    best_match = known_name
        if best_score >= 0.99:
            logger.info(f"Fuzzy matched '{item_name}' to '{best_match}' with score {best_score:.2f}")
            return best_match
        return None

    def guess_item_category(self, item_name):
        item_lower = item_name.lower()
        category_keywords = {
            "Welcome_Drinks": ["punch", "Packed Juice","fresh fruit juice", "juice", "mojito", "drinks","milk", "tea", "coffee", "juice", "butter milk", "lassi", "soda",  "water melon juice"],
            "Appetizers": ["tikka", "65","Sauteed Grilled Chicken Sausage", "paneer","Fish Fingers","Mashed Potatos","Cheese Fries","french fires","Potato Skins","Pepper Chicken (Oil Fry)","Lemon Chicken","kabab", "hariyali kebab", "tangdi", "drumsticks", "nuggets","majestic","roll", "poori", "masala vada", "alasanda vada", "veg bullets", "veg spring rolls", "hara bara kebab", "kebab", "lollipop", "chicken lollipop", "pakora", "kodi", "cut", "bajji", "vepudu", "roast", "kurkure", "afghani kebab", "corn", "manchuria", "manchurian", "gobi","jalapeno pop up","Chilli Garlic "],
            "Soups": ["soup", "shorba","mutton marag","broth","cream of chicken","paya","hot and sour",],
            "Fried": ["fried rice"],
            "Italian": ["pasta", "noodles", "white pasta", "veg garlic soft noodles","macroni"],
            "Fry": ["fry", "bendi kaju","Dondakaya Fry","Bhindi Fry","Aloo Fry","Cabbage Fry"],
            "Liquids(Less_Dense)": ["rasam","Pachi pulusu","Sambar", "charu", "majjiga pulusu", "Miriyala Rasam","chintapandu rasam","lemon pappucharu","mulakaya pappucharu"],
            "Liquids(High_Dense)": ["ulavacharu"],
            "Curries": ["iguru","Paneer Butter Masala","Chicken Chettinad","gutti vankaya curry","kadai","scrambled egg curry","baigan bartha","bendi do pyaza","boiled egg cury","chana masala","curry", "gravy", "masala", "kurma", "butter","pulusu","mutton rogan josh curry","kadai", "tikka masala", "dal tadka", "boti", "murgh", "methi", "bhurji", "chatapata", "pulsu", "vegetable curry", "dum aloo curry"],
            "Rice": ["steamed rice", "kaju ghee rice", "bagara rice"],
            "Flavored_Rice": ["muddapappu avakai annam","Ragi Sangati", "pudina rice","temple style pulihora", "pappucharu annam","pappu charu annam","cocount rice","cocunut rice","pulihora", "curd rice", "jeera rice", "gongura rice", "Muddapappu Avakaya Annam", "sambar rice", "Muddapappu avakai Annam", "annam"],
            "Pulav": ["pulav","Mutton Fry Piece Pulav","Natukodi Pulav","jeera mutter pulav", "fry piece pulav", "ghee pulav","Green Peas Pulav","Meal Maker Pulav","Paneer Pulav"],
            "Biryani": ["biryani", "biriyani", "Mutton Kheema Biryani","biriani", "panaspottu biryani","egg biryani","chicken chettinad biryani","ulavacharu chicken biryani", "mushroom biryani", "veg biryani", "chicken dum biryani"],
            "Breads": ["naan", "paratha", "kulcha", "pulka", "chapati", "rumali roti","laccha paratha","masala kulcha","panner kulcha","butter garlic naan","roti,pudina naan","tandoori roti"],
            "Dal": ["dal", "lentil", "pappu", "Mamidikaya Pappu (Mango)","Dal Makhani","Dal Tadka","sorakaya pappu", "thotakura pappu", "tomato pappu", "yellow dal""chintakaya papu","palakura pappu","thotakura pappu","tomato pappu","yellow dal tadka"],
            "Chutney":["peanut chutney","allam chutney","green chutney","pudina chutney","dondakay chutney"],
            "Ghee": ["ghee"],
            "Podi": ["podi"],
            "Pickle": ["pickle"],
            "Paan": ["paan", "pan"],
            "Dips": ["dip","Sour cream Dip","jam","Tandoori Mayo","Mayonnaise Dip","Hummus Dip","Garlic Mayonnaise Dip"],
            "Roti_Pachadi": ["Beerakaya Pachadi", "roti pachadi", "Tomato Pachadi","Vankaya Pachadi","Roti Pachadi", "pachadi","gongura onion chutney"],
            "Crispers": ["fryums", "papad", "crispers"],
            "Raitha": ["raitha", "Raitha", "boondi raitha"],
            "Salan": ["salan", "Salan", "Mirchi Ka Salan"],
            "Fruits": ["seaonsal", "mixed", "cut", "fruit", "Fruit"],
            "Salad": ["salad", "Salad", "ceasar", "green salad", "boiled peanut salad","boiled peanut salad","mexican corn salad","Laccha Pyaaz","Cucumber Salad"],
            "Curd": ["curd", "set curd"],
            "Desserts": ["brownie", "walnut brownie", "Gajar Ka Halwa","Chocolate Brownie","Assorted Pastries","halwa","Semiya Payasam (Kheer)","Sabudana Kheer","Kesari Bath","Double Ka Meetha", "carrot halwa", "shahi ka tukda", "gulab jamun", "apricot delight", "baked gulab jamun", "bobbattu", "bobbatlu", "kalajamun", "rasagulla", "laddu", "poornam", "apricot delight", "gulab jamun", "rasammaiah"],
            "Breakfast": ["idly", "dosa", "vada", "upma","Rava Khichdi","Bisi Bela Bath","Sabudana Khichdi","Upma","Millet Upma", "pongal", "mysore bonda", "idly"],
            "Sandwich": ["sandwich","Veg Sandwitch"],
            "Cup_Cakes": ["cup cake", "cupcake"]
        }
        for category, keywords in category_keywords.items():
            if item_lower in keywords:
                logger.info(f"Categorized '{item_name}' as '{category}' based on exact match")
                return category
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in item_lower:
                    logger.info(f"Categorized '{item_name}' as '{category}' based on keyword '{keyword}'")
                    return category
        logger.warning(f"Using default category 'Curries' for '{item_name}'")
        return "Curries"

    def determine_item_properties(self, item_name):
        """
        More robust determination of item properties using metadata and pattern matching.
        """
        # Handle '>' prefix automatically
        clean_item_name = item_name.replace('> ', '').replace('>', '') if isinstance(item_name, str) else item_name

        # Start with defaults
        properties = {
            "category": None,
            "unit": "g",
            "is_veg": True
        }

        # Try direct match in metadata first (most reliable)
        if clean_item_name in self.item_metadata:
            return {
                "category": self.item_metadata[clean_item_name].category,
                "unit": self.item_metadata[clean_item_name].unit,
                "is_veg": getattr(self.item_metadata[clean_item_name], "is_veg", True)
            }

        # Try finding closest item in metadata using enhanced matcher
        mapped_item, _ = self.item_matcher.find_item(clean_item_name, self.item_metadata) if hasattr(self,
                                                                                                     'item_matcher') else (
            None, None)
        if mapped_item and mapped_item in self.item_metadata:
            return {
                "category": self.item_metadata[mapped_item].category,
                "unit": self.item_metadata[mapped_item].unit,
                "is_veg": getattr(self.item_metadata[mapped_item], "is_veg", True)
            }

        # If no match in metadata, use enhanced category detection
        item_lower = clean_item_name.lower() if isinstance(clean_item_name, str) else ""
        properties["category"] = self.guess_item_category(clean_item_name)

        # Enhanced unit detection based on both category and item name patterns
        if properties["category"] in self.food_rules.category_rules:
            rule = self.food_rules.category_rules[properties["category"]]
            default_qty = rule.get("default_quantity", "0g")
            unit = self.food_rules.extract_unit(default_qty)
            if unit:
                properties["unit"] = unit

        # Specific patterns for determining unit type
        # Desserts with piece-based units
        if properties["category"] == "Desserts":
            piece_patterns = ["jamun", "gulab", "rasgulla", "laddu", "burfi", "jalebi", "poornam", "buralu",
                              "delight", "mysore pak", "badusha"]
            if any(pattern in item_lower for pattern in piece_patterns):
                properties["unit"] = "pcs"

        # Appetizers with piece-based units
        elif properties["category"] == "Appetizers":
            piece_patterns = ["samosa", "tikka", "kebab", "roll", "cutlet", "patty", "vada", "bonda",
                              "pakora", "spring roll"]
            if any(pattern in item_lower for pattern in piece_patterns):
                properties["unit"] = "pcs"

        # Breakfast items with piece-based units
        elif properties["category"] == "Breakfast":
            piece_patterns = ["idli", "vada", "dosa", "uttapam", "poori", "paratha", "sandwich", "bun"]
            if any(pattern in item_lower for pattern in piece_patterns):
                properties["unit"] = "pcs"

        # Breads are always pieces
        elif properties["category"] in ["Breads", "Bread"]:
            properties["unit"] = "pcs"

        # Liquids are ml
        elif "Liquids" in properties["category"] or properties["category"] in ["Welcome_Drinks", "Soups"]:
            properties["unit"] = "ml"

        # Determine veg/non-veg status with more patterns
        non_veg_indicators = ["chicken", "mutton", "fish", "prawn", "beef", "pork", "egg", "meat",
                              "non veg", "kodi", "murg", "lamb", "goat", "seafood", "keema", "crab"]
        if any(indicator in item_lower for indicator in non_veg_indicators):
            properties["is_veg"] = False

        return properties

    def build_menu_context(self, event_time, meal_type, event_type, guest_count, selected_items):
        """Build comprehensive menu context to enhance prediction accuracy"""
        # Initialize context
        menu_context = {
            'categories': [],
            'items': selected_items,
            'total_items': len(selected_items),
            'meal_type': meal_type,
            'event_time': event_time,
            'event_type': event_type,
            'guest_count': guest_count,
            'items_by_category': {},
            'item_properties': {},
            'special_categories': ["Appetizers", "Desserts", "Breakfast"],
            'items_by_unit_type': {}
        }

        # Process each item to determine properties and categories
        for item in selected_items:
            # Get item properties
            properties = self.determine_item_properties(item)
            category = properties["category"]
            unit = properties["unit"]

            # Store item properties
            menu_context['item_properties'][item] = properties

            # Organize by category
            if category not in menu_context['items_by_category']:
                menu_context['items_by_category'][category] = []
                menu_context['categories'].append(category)
            menu_context['items_by_category'][category].append(item)

            # Organize by category and unit type (for special categories)
            if category in menu_context['special_categories']:
                if category not in menu_context['items_by_unit_type']:
                    menu_context['items_by_unit_type'][category] = {'g': [], 'ml': [], 'pcs': []}
                menu_context['items_by_unit_type'][category][unit].append(item)

        # Add context flags for dependencies
        menu_context['has_biryani'] = 'Biryani' in menu_context['categories']
        menu_context['has_rice'] = 'Rice' in menu_context['categories']
        menu_context['has_curries'] = 'Curries' in menu_context['categories']
        menu_context['has_dal'] = 'Dal' in menu_context['categories']

        return menu_context







    def build_item_name_mapping(self, data):
        logger.info("Building item_name mapping")
        self.item_name_mapping = {}
        item_groups = data.groupby(['Item_name'])
        for item_name, _ in item_groups:
            std_name = self.standardize_item_name(item_name)
            if std_name:
                self.item_name_mapping[std_name] = item_name
                self.item_name_mapping[item_name] = item_name
        logger.info(f"Built mapping for {len(self.item_name_mapping)} item variations")

    def build_item_metadata(self, data):
        logger.info("Building item metadata from training data")
        if not hasattr(self, 'item_name_mapping'):
            self.build_item_name_mapping(data)
        item_groups = data.groupby(['Item_name'])
        for item_name, group in item_groups:
            # Ensure item_name is a string, not a tuple
            if isinstance(item_name, tuple) and len(item_name) > 0:
                item_name = item_name[0]

            category = group['Category'].iloc[0]
            unit = self.extract_unit(group['Per_person_quantity'].iloc[0])
            if not unit:
                unit = self.food_rules.infer_default_unit(category)
            unit = self.food_rules.validate_unit(category, unit)
            self.item_metadata[item_name] = ItemMetadata(category=category, unit=unit)
            std_name = self.standardize_item_name(item_name)
            if std_name != item_name:
                self.item_metadata[std_name] = ItemMetadata(category=category, unit=unit)

        logger.info(f"Built metadata for {len(self.item_metadata)} items")

        # Debug which items contain "sambar"
        sambar_items = [key for key in self.item_metadata.keys()
                        if isinstance(key, str) and "sambar" in key.lower()]
        print(f"DEBUG: Items with 'sambar' in name: {sambar_items}")

        self.item_matcher = FoodItemMatcher(self.item_metadata)
        logger.info("Initialized FoodITemMatcher with item metadata")

    def fit(self, data):
        logger.info("Starting model training")
        if not self.item_metadata:
            self.build_item_metadata(data)

        data['quantity_value'] = data['Per_person_quantity'].apply(self.extract_quantity_value)
        X = self.prepare_features(data)

        categories = data['Category'].unique()
        for category in categories:
            logger.info(f"Training model for category: {category}")
            category_data = data[data['Category'] == category]
            if category_data.empty:
                continue
            category_data_subset = category_data[['Order_Number', 'quantity_value', 'Guest_Count']]
            category_per_person = category_data_subset.groupby('Order_Number').agg({
                'quantity_value': 'sum',
                'Guest_Count': 'first'
            }).reset_index()
            category_per_person = category_per_person.rename(columns={'quantity_value': 'per_person_quantity'})
            category_per_person = category_per_person[['Order_Number', 'per_person_quantity']]
            category_train = pd.merge(
                category_per_person,
                data[['Order_Number', 'Event_Time', 'Meal_Time', 'Event_Type']].drop_duplicates(),
                on='Order_Number',
                how='left'
            )
            if category_train.empty:
                continue
            if category not in self.category_models:
                self.category_models[category] = XGBRegressor(
                    n_estimators=500, learning_rate=0.1, max_depth=6, objective='reg:squarederror', random_state=42
                )
                self.category_scalers[category] = RobustScaler()
            X_cat = self.prepare_features(category_train)
            X_cat_scaled = self.category_scalers[category].fit_transform(X_cat)
            self.category_models[category].fit(X_cat_scaled, category_train['per_person_quantity'])

        training_items = set(data['Item_name'].unique())
        items_to_train = {item_name: metadata for item_name, metadata in self.item_metadata.items() if item_name in training_items}
        for item_name, metadata in items_to_train.items():
            logger.info(f"Training model for item: {item_name}")
            item_data = data[data['Item_name'] == item_name]
            item_data_subset = item_data[['Order_Number', 'quantity_value', 'Guest_Count']]
            item_per_person = item_data_subset.groupby('Order_Number').agg({
                'quantity_value': 'sum',
                'Guest_Count': 'first'
            }).reset_index()
            item_per_person = item_per_person.rename(columns={'quantity_value': 'per_person_quantity'})
            item_per_person = item_per_person[['Order_Number', 'per_person_quantity']]
            item_train = pd.merge(
                item_per_person,
                data[['Order_Number', 'Event_Time', 'Meal_Time', 'Event_Type']].drop_duplicates(),
                on='Order_Number',
                how='left'
            )
            if item_train.empty:
                continue
            if item_name not in self.item_models:
                self.item_models[item_name] = XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
                    reg_lambda=1, reg_alpha=0.5, random_state=42
                )
                self.item_scalers[item_name] = RobustScaler()
            X_item = self.prepare_features(item_train)
            X_item_scaled = self.item_scalers[item_name].fit_transform(X_item)
            self.item_models[item_name].fit(X_item_scaled, item_per_person['per_person_quantity'])
        logger.info("Model training completed")

    def identify_main_items(self, menu_context):
        """
        Identify the main items in the menu (Biryani, Flavored Rice, Steamed Rice)

        Args:
            menu_context: The menu context dictionary built by build_menu_context

        Returns:
            Dictionary with main item categories found and their specific items
        """
        main_categories = ["Biryani", "Flavored_Rice", "Rice"]
        found_main_items = {
            "has_main_item": False,
            "categories_found": [],
            "items_by_category": {},
            "primary_main_item": None
        }

        # Check if any main categories exist in the menu
        for category in main_categories:
            if category in menu_context['categories'] or category in menu_context['items_by_category']:
                found_main_items["has_main_item"] = True
                found_main_items["categories_found"].append(category)

                # Get items in this category
                if category in menu_context['items_by_category']:
                    found_main_items["items_by_category"][category] = menu_context['items_by_category'][category]

        # Determine primary main item based on priority: Biryani > Flavored_Rice > Rice
        for priority_category in main_categories:
            if priority_category in found_main_items["categories_found"]:
                found_main_items["primary_main_category"] = priority_category
                if found_main_items["items_by_category"].get(priority_category):
                    found_main_items["primary_main_item"] = found_main_items["items_by_category"][priority_category][0]
                break

        return found_main_items

    def predict(self, event_time, meal_type, event_type, guest_count, selected_items):
        logger.info(f"Making prediction for event: {event_type}, {event_time}, {meal_type}, {guest_count} guests")

        # Build menu context - NEW FEATURE
        menu_context = self.build_menu_context(event_time, meal_type, event_type, guest_count, selected_items)

        # Identify main items in the menu - NEW FEATURE
        main_items_info = self.identify_main_items(menu_context)
        logger.info(f"Main items identified: {main_items_info['categories_found']}")
        if main_items_info.get('primary_main_item'):
            logger.info(f"Primary main item: {main_items_info['primary_main_item']}")

        input_data = pd.DataFrame({'Event_Time': [event_time], 'Meal_Time': [meal_type], 'Event_Type': [event_type]})
        X = self.prepare_features(input_data)
        predictions = {}
        items_by_category = {}
        unmapped_items = []
        original_to_mapped = {}
        total_items = len(selected_items)

        for item in selected_items:
            std_item = self.standardize_item_name(item)
            mapped_item = getattr(self, 'item_name_mapping', {}).get(std_item, None)
            if not mapped_item:
                mapped_item = self.find_closest_item(item)
                if mapped_item:
                    self.item_name_mapping[std_item] = mapped_item
                    # self.item_name_mapping[mapped_item] = mapped_item
            if mapped_item and mapped_item in self.item_metadata:
                category = self.item_metadata[mapped_item].category
                if category not in items_by_category:
                    items_by_category[category] = []
                items_by_category[category].append(mapped_item)
                original_to_mapped[item] = mapped_item
                if item != mapped_item:
                    logger.info(f"Mapped '{item}' to known item '{mapped_item}'")
            else:
                unmapped_items.append(item)
                logger.warning(f"No match found for item: {item}")

        if unmapped_items:
            for item in unmapped_items:
                category = self.guess_item_category(item)
                if category not in items_by_category:
                    items_by_category[category] = []
                items_by_category[category].append(item)
                original_to_mapped[item] = item
                unit = self.food_rules.infer_default_unit(category)
                default_qty, default_unit = self.food_rules.get_default_quantity(category)
                if default_unit:
                    unit = default_unit
                std_item = self.standardize_item_name(item)
                self.item_metadata[item] = ItemMetadata(category=category, unit=unit)
                self.item_metadata[std_item] = ItemMetadata(category=category, unit=unit)

        category_per_person = {}
        category_quantities = {}
        item_quantities = {}

        for category, items in items_by_category.items():
            if category in self.category_models:
                X_cat_scaled = self.category_scalers[category].transform(X)
                ml_prediction = float(self.category_models[category].predict(X_cat_scaled)[0])
                qty, unit = self.food_rules.apply_category_rules(category, guest_count, len(items),
                                                                 total_items=total_items)
                if qty > 0:
                    category_per_person[category] = qty
                    category_quantities[category] = {"value": qty, "unit": unit}
                else:
                    category_per_person[category] = ml_prediction
                    unit = self.item_metadata[items[0]].unit if items else "g"
                    category_quantities[category] = {"value": ml_prediction, "unit": unit}
            else:
                qty, unit = self.food_rules.get_default_quantity(category)
                category_per_person[category] = qty if qty > 0 else 0.0
                category_quantities[category] = {"value": qty if qty > 0 else 0.0, "unit": unit}

            for item in items:
                qty, unit = self.food_rules.get_item_specific_quantity(item, guest_count, self.item_specific_data)
                if qty is not None and unit is not None:
                    item_quantities[item] = {"value": qty, "unit": unit}

        # NEW FEATURE: Apply main item adjustments for both the main item itself and complementary items
        # NEW FEATURE: Apply main item adjustments based on total items and main item presence
        if main_items_info["has_main_item"] and main_items_info.get("primary_main_category"):
            primary_main = main_items_info["primary_main_category"]
            total_menu_items = len(selected_items)  # Get total number of items in the order

            # FIRST, adjust the main item quantity itself based on its role in the meal and total items
            if primary_main == "Rice" and "Rice" in category_quantities:
                # Check if this is the only main item (no Biryani or Flavored_Rice)
                if "Biryani" not in items_by_category and "Flavored_Rice" not in items_by_category:
                    orig_qty = category_quantities["Rice"]["value"]
                    # Apply quantity based on total items
                    if total_menu_items <= 4:
                        adjusted_qty = max(orig_qty, 400.0)  # At least 400g for small menus
                    else:
                        adjusted_qty = orig_qty  # Keep original predicted value for larger menus

                    category_quantities["Rice"]["value"] = adjusted_qty
                    logger.info(
                        f"Main item adjustment: Adjusted Rice quantity based on menu size ({total_menu_items} items) from {orig_qty:.2f} -> {adjusted_qty:.2f}")
                    # Directly update category_per_person as well
                    if "Rice" in category_per_person:
                        category_per_person["Rice"] = adjusted_qty

            elif primary_main == "Biryani" and "Biryani" in category_quantities:
                # For Biryani as primary main, adjust quantity based on total menu items
                orig_qty = category_quantities["Biryani"]["value"]

                # Only apply special rules when Biryani is the main rice item
                if "Rice" not in items_by_category and "Flavored_Rice" not in items_by_category:
                    # Apply quantity based on total items
                    if total_menu_items <= 3:  # Small menu with few items
                        adjusted_qty = max(orig_qty, 550.0)  # Increase to 550g for small menus
                    else:
                        adjusted_qty = orig_qty  # Keep original predicted value for larger menus

                    category_quantities["Biryani"]["value"] = adjusted_qty
                    logger.info(
                        f"Main item adjustment: Adjusted Biryani quantity based on menu size ({total_menu_items} items) from {orig_qty:.2f} -> {adjusted_qty:.2f}")
                    # Directly update category_per_person as well
                    if "Biryani" in category_per_person:
                        category_per_person["Biryani"] = adjusted_qty

            elif primary_main == "Flavored_Rice" and "Flavored_Rice" in category_quantities:
                # For Flavored Rice as primary main, ensure minimum quantity
                orig_qty = category_quantities["Flavored_Rice"]["value"]

                # Only apply special rules when Flavored Rice is the main rice item
                if "Rice" not in items_by_category and "Biryani" not in items_by_category:
                    # Apply quantity based on total items
                    if total_menu_items <= 5:  # Small menu with few items
                        adjusted_qty = max(orig_qty, 250.0)  # Increase to 250g for small menus
                    else:
                        adjusted_qty = orig_qty  # Keep original predicted value for larger menus

                    category_quantities["Flavored_Rice"]["value"] = adjusted_qty
                    logger.info(
                        f"Main item adjustment: Adjusted Flavored Rice quantity based on menu size ({total_menu_items} items) from {orig_qty:.2f} -> {adjusted_qty:.2f}")
                    # Directly update category_per_person as well
                    if "Flavored_Rice" in category_per_person:
                        category_per_person["Flavored_Rice"] = adjusted_qty

            # THEN adjust quantities for items that complement the main item
            for category in category_quantities:
                # Skip the main category itself
                if category == primary_main:
                    continue

                # Apply main-item specific adjustments
                if primary_main == "Biryani":
                    # Adjust quantities for items that pair with Biryani
                    if category == "Raitha":
                        orig_qty = category_quantities[category]["value"]
                        # Increase raitha quantity for biryani
                        adjusted_qty = orig_qty * 1.2
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Increased {category} quantity for Biryani ({orig_qty:.2f} -> {adjusted_qty:.2f})")
                    elif category == "Salan":
                        orig_qty = category_quantities[category]["value"]
                        # Increase salan quantity for biryani
                        adjusted_qty = orig_qty * 1.15
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Increased {category} quantity for Biryani ({orig_qty:.2f} -> {adjusted_qty:.2f})")
                    elif category in ["Rice", "Flavored_Rice"]:
                        # Reduce other rice quantities when biryani is present
                        orig_qty = category_quantities[category]["value"]
                        adjusted_qty = orig_qty * 0.7
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Reduced {category} quantity due to Biryani ({orig_qty:.2f} -> {adjusted_qty:.2f})")

                elif primary_main == "Flavored_Rice":
                    # Adjust quantities for items that pair with Flavored Rice
                    if category == "Rice":
                        # Reduce plain rice when flavored rice is present
                        orig_qty = category_quantities[category]["value"]
                        adjusted_qty = orig_qty * 0.7
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Reduced {category} quantity due to Flavored Rice ({orig_qty:.2f} -> {adjusted_qty:.2f})")
                    elif category == "Curries":
                        # Might want more curry with flavored rice
                        orig_qty = category_quantities[category]["value"]
                        adjusted_qty = orig_qty * 1.1
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Increased {category} quantity for Flavored Rice ({orig_qty:.2f} -> {adjusted_qty:.2f})")

                elif primary_main == "Rice":
                    # Adjust quantities for items that pair with plain Rice
                    if category == "Curries":
                        # More curry with plain rice
                        orig_qty = category_quantities[category]["value"]
                        adjusted_qty = orig_qty * 1.2
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Increased {category} quantity for Rice ({orig_qty:.2f} -> {adjusted_qty:.2f})")
                    elif category == "Dal":
                        # More dal with plain rice
                        orig_qty = category_quantities[category]["value"]
                        adjusted_qty = orig_qty * 1.15
                        category_quantities[category]["value"] = adjusted_qty
                        logger.info(
                            f"Main item adjustment: Increased {category} quantity for Rice ({orig_qty:.2f} -> {adjusted_qty:.2f})")

        # Continue with existing logic
        for category, items in items_by_category.items():
            dependent_categories = self.food_rules.get_dependent_categories(category)
            for dep in dependent_categories:
                dep_normalized = dep.strip()
                matching_categories = [c for c in items_by_category.keys() if c.lower() == dep_normalized.lower()]
                if matching_categories:
                    dep_category = matching_categories[0]
                    orig_qty = category_per_person[category]
                    unit = category_quantities[category]["unit"]
                    adjusted_qty = self.food_rules.apply_dependency_rules(category, dep_category, orig_qty, unit)
                    category_per_person[category] = adjusted_qty
                    category_quantities[category]["value"] = adjusted_qty

        for category in category_per_person:
            orig_qty = category_per_person[category]
            modified_qty = self.food_rules.apply_meal_type_modifier(meal_type, orig_qty)
            category_per_person[category] = modified_qty
            category_quantities[category]["value"] = modified_qty

        item_per_person = {}
        for category, items in items_by_category.items():
            if category in category_per_person:
                category_qty = category_per_person[category]
                category_unit = category_quantities[category]["unit"]
                for item in items:
                    if item in item_quantities:
                        item_per_person[item] = item_quantities[item]["value"] / guest_count
                    else:
                        item_per_person[item] = category_qty

        def auto_convert(quantity: float, unit: str) -> Tuple[float, str]:
            unit_lower = unit.lower()
            if unit_lower == "g" and quantity >= 1000:
                return quantity / 1000, "kg"
            elif unit_lower == "ml" and quantity >= 1000:
                return quantity / 1000, "l"
            return quantity, unit

        for orig_item in selected_items:
            mapped_item = original_to_mapped.get(orig_item, orig_item)
            if mapped_item in item_per_person:
                per_person_qty = item_per_person[mapped_item]
                category = self.item_metadata[mapped_item].category
                if mapped_item in item_quantities:
                    unit = item_quantities[mapped_item]["unit"]
                    total_qty = item_quantities[mapped_item]["value"]
                else:
                    unit = category_quantities.get(category, {}).get("unit", "g")
                    unit = self.food_rules.validate_unit(category, unit)
                    total_qty = per_person_qty * guest_count
                converted_total_qty, converted_total_unit = auto_convert(total_qty, unit)

                # Calculate per-person quantity in a user-friendly unit
                display_per_person_qty = per_person_qty
                display_per_person_unit = unit
                if unit == "kg":
                    display_per_person_qty = per_person_qty * 1000  # Convert kg to g
                    display_per_person_unit = "g"
                elif unit == "l":
                    display_per_person_qty = per_person_qty * 1000  # Convert l to ml
                    display_per_person_unit = "ml"

                predictions[orig_item] = {
                    "total": f"{converted_total_qty}{converted_total_unit}",
                    "per_person": f"{display_per_person_qty:.2f}{display_per_person_unit}",
                    "unit": unit,  # Store the original unit
                    "converted_unit": converted_total_unit  # Store the converted unit for pricing
                }
            else:
                logger.warning(f"Item {orig_item} missing from predictions, applying fallback")
                if mapped_item in self.item_metadata:
                    category = self.item_metadata[mapped_item].category
                    if category in category_per_person:
                        per_person_qty = category_per_person[category]
                        unit = category_quantities.get(category, {}).get("unit", "g")
                        unit = self.food_rules.validate_unit(category, unit)
                        total_qty = per_person_qty * guest_count
                        converted_total_qty, converted_total_unit = auto_convert(total_qty, unit)

                        # Calculate per-person quantity in a user-friendly unit
                        display_per_person_qty = per_person_qty
                        display_per_person_unit = unit
                        if unit == "kg":
                            display_per_person_qty = per_person_qty * 1000  # Convert kg to g
                            display_per_person_unit = "g"
                        elif unit == "l":
                            display_per_person_qty = per_person_qty * 1000  # Convert l to ml
                            display_per_person_unit = "ml"

                        predictions[orig_item] = {
                            "total": f"{converted_total_qty}{converted_total_unit}",
                            "per_person": f"{display_per_person_qty:.2f}{display_per_person_unit}",
                            "unit": unit,
                            "converted_unit": converted_total_unit
                        }
        return predictions

    def calculate_price(self, converted_qty, category, guest_count, item_name, unit):
        normalized_category = self.food_rules.normalize_category_name(category)
        rule = self.food_rules.category_rules.get(normalized_category, {})

        # Parameters
        FC = 6000  # Fixed cost
        Qmin_static = 50  # Static minimum quantity
        beta = 0.5  # Exponent for dynamic Qmin

        # Get item-specific data
        std_item_name = self.standardize_item_name(item_name)
        item_data = self.item_specific_data.get(std_item_name, {})
        preferred_unit = item_data.get('preferred_unit', None)
        base_price_per_piece = item_data.get('base_price_per_piece', None)
        base_price_per_kg = item_data.get('base_price_per_kg', None)

        # Get item-specific VC and p_value
        vc_data = self.item_vc_mapping.get(std_item_name, {})
        VC = vc_data.get('VC', rule.get("vc_price", 220))  # Fallback to category VC
        p_value = vc_data.get('p_value', 0.19)  # Fallback to 0.19

        # If base_price_per_kg is not provided, use VC
        if base_price_per_kg is None:
            base_price_per_kg = VC
            logger.info(f"No base_price_per_kg for {item_name}, using VC={VC} as base_price_per_kg")

        # Use the provided unit, validate it
        unit = self.food_rules.validate_unit(normalized_category, unit)
        if preferred_unit:
            unit = preferred_unit

        # Calculate total quantity (Q) correctly
        converted_qty = float(converted_qty)
        Q = converted_qty  # Total quantity (already in the correct unit: pcs, g, kg, etc.)

        # Convert Q to the threshold unit (e.g., kg for g, l for ml)
        if unit == "pcs":
            Q_threshold = 45
            Q_in_threshold_unit = Q
            base_price_per_unit = base_price_per_piece if base_price_per_piece is not None else VC
        else:
            if unit == "kg":
                Q_threshold = 50
                Q_in_threshold_unit = Q
            elif unit == "g":
                Q_threshold = 50
                Q_in_threshold_unit = Q / 1000  # Convert g to kg
            elif unit == "l":
                Q_threshold = 60
                Q_in_threshold_unit = Q
            elif unit == "ml":
                Q_threshold = 60
                Q_in_threshold_unit = Q / 1000  # Convert ml to l
            else:
                Q_threshold = 50
                Q_in_threshold_unit = Q / 1000
            base_price_per_unit = base_price_per_kg

        # Calculate Qmin
        Qmin = max(Qmin_static, Q_in_threshold_unit ** beta)
        base_rate = FC / Qmin + VC

        # Base price per unit (in threshold unit, e.g., kg for g, l for ml)
        if unit != "pcs":
            base_price_per_unit = np.where(
                Q_in_threshold_unit <= Q_threshold,
                base_rate * Q_in_threshold_unit**(-p_value),
                base_rate * Q_threshold**(-p_value)
            )

        # Calculate total price
        if Q_in_threshold_unit <= Q_threshold:
            total_price = base_price_per_unit * Q_in_threshold_unit
        else:
            total_price = (base_price_per_unit * Q_threshold) + (base_price_per_unit * (Q_in_threshold_unit - Q_threshold))

        # Calculate per-person quantity and per-person price
        # Adjust per_person_quantity to be in the threshold unit (e.g., kg for g)
        if unit == "g":
            per_person_quantity = (Q / 1000) / guest_count if guest_count > 0 else 0  # Convert g to kg
        elif unit == "ml":
            per_person_quantity = (Q / 1000) / guest_count if guest_count > 0 else 0  # Convert ml to l
        else:
            per_person_quantity = Q / guest_count if guest_count > 0 else 0  # Already in threshold unit (kg, l, pcs)

        per_person_price = per_person_quantity * base_price_per_unit

        print(f"Debug: item={item_name}, category={normalized_category}, unit={unit}, FC={FC}, VC={VC}, p={p_value}")
        #print(f"Debug: Q={Q}, Q_in_threshold_unit={Q_in_threshold_unit}, Qmin={Qmin}, base_price_per_unit={base_price_per_unit}, Q_threshold={Q_threshold}")
        #print(f"Debug: total_price={total_price}, per_person_quantity={per_person_quantity}, per_person_price={per_person_price}")

        return total_price, base_price_per_unit, per_person_price

    def get_predictions_from_terminal(self):
        """Get predictions by taking input from the terminal."""
        print("\n=== Food Quantity and Price Prediction System ===")

        # Event Time
        print("\nAvailable Event Times: Morning, Afternoon, Evening, Night")
        event_time = input("Enter Event Time: ").strip().capitalize()
        valid_event_times = ["Morning", "Afternoon", "Evening", "Night"]
        while event_time not in valid_event_times:
            print("Invalid Event Time. Please choose from: Morning, Afternoon, Evening, Night")
            event_time = input("Enter Event Time: ").strip().capitalize()

        # Meal Type
        print("\nAvailable Meal Types: Breakfast, Lunch, Hi-Tea, Dinner")
        meal_type = input("Enter Meal Type: ").strip().capitalize()
        valid_meal_types = ["Breakfast", "Lunch", "Hi-Tea", "Dinner"]
        while meal_type not in valid_meal_types:
            print("Invalid Meal Type. Please choose from: Breakfast, Lunch, Hi-Tea, Dinner")
            meal_type = input("Enter Meal Type: ").strip().capitalize()

        # Event Type
        print("\nAvailable Event Types: Wedding, Birthday Party, Corporate Event, Casual Gathering")
        event_type = input("Enter Event Type: ").strip()
        valid_event_types = ["Wedding", "Birthday Party", "Corporate Event", "Casual Gathering"]
        while event_type not in valid_event_types:
            print("Invalid Event Type. Please choose from: Wedding, Birthday Party, Corporate Event, Casual Gathering")
            event_type = input("Enter Event Type: ").strip()

        # Number of Guests
        while True:
            try:
                guest_count = int(input("\nEnter Number of Guests (positive integer): "))
                if guest_count <= 0:
                    print("Number of guests must be a positive integer.")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer for the number of guests.")

        # Menu Items
        print("\nEnter Menu Items (one per line). Press Enter twice to finish:")
        menu_items = []
        while True:
            item = input().strip()
            if item == "":
                break
            menu_items.append(item)

        if not menu_items:
            print("No menu items provided. Exiting.")
            return

        # Get predictions
        predictions = self.predict(event_time, meal_type, event_type, guest_count, menu_items)

        # Process and display results
        print("\n=== Predictions ===")
        print(f"{'Items':<20} {'Quantity':<15} {'Per Person Weight':<20} {'Per Person Price':<20} {'Total Price':<15}")
        print("-" * 90)

        total_per_person_weight_g = 0.0  # For solid items (in grams, excluding pcs)
        total_per_person_volume_ml = 0.0  # For liquid items (in milliliters)
        total_per_person_pieces = 0.0  # For items in pcs

        for item, qty_data in predictions.items():
            total_qty = qty_data['total']
            per_person_qty_str = qty_data['per_person']
            per_person_qty = self.extract_quantity_value(per_person_qty_str)
            unit = self.extract_unit(per_person_qty_str)

            # Determine the category to check if it's a liquid item
            std_item = self.standardize_item_name(item)
            mapped_item = getattr(self, 'item_name_mapping', {}).get(std_item, item)
            category = self.item_metadata[mapped_item].category if mapped_item in self.item_metadata else self.guess_item_category(item)

            # Check if the item is a liquid
            liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
            is_liquid = category in liquid_categories

            # Accumulate totals for per-person plate
            if unit == "pcs":
                total_per_person_pieces += per_person_qty
            elif is_liquid:
                if unit == "ml":
                    total_per_person_volume_ml += per_person_qty
                elif unit == "l":
                    total_per_person_volume_ml += per_person_qty * 1000
            else:
                if unit == "g":
                    total_per_person_weight_g += per_person_qty
                elif unit == "kg":
                    total_per_person_weight_g += per_person_qty * 1000

            # Calculate price using the converted unit
            total_qty_val = self.extract_quantity_value(total_qty)
            converted_unit = qty_data['converted_unit']  # Use the converted unit
            total_price, base_price_per_unit, per_person_price = self.calculate_price(total_qty_val, category, guest_count, item, unit=converted_unit)

            # Display the row
            print(f"{item:<20} {total_qty:<15} {per_person_qty_str:<20} ₹{per_person_price:<19.2f} ₹{total_price:<14.2f}")

        # Display total per-person plate
        print("\n=== Total Per-Person Plate ===")
        if total_per_person_weight_g > 0:
            print(f"Total Items: {total_per_person_weight_g:.2f}g")
        if total_per_person_volume_ml > 0:
            print(f"Liquid Items: {total_per_person_volume_ml:.2f}ml")
        if total_per_person_pieces > 0:
            print(f"Pieces: {total_per_person_pieces:.2f}pcs")

def load_and_train_model(data_path, item_vc_file="Book1.xlsx", item_data_file="item_data.csv", category_constraints=None):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_excel(data_path)
    predictor = HierarchicalFoodPredictor(category_constraints=category_constraints, item_vc_file=item_vc_file, item_data_file=item_data_file)
    predictor.fit(data)
    return predictor

def main():
    """Main function to run the predictor from the terminal."""
    # Load and train the model
    try:
        predictor = load_and_train_model(r"C:\Users\Syed Ashfaque Hussai\OneDrive\Desktop\CraftMyplate Machine Learning Task\DB211.xlsx",
                                         item_vc_file=r"C:\Users\Syed Ashfaque Hussai\OneDrive\Desktop\CraftMyplate Machine Learning Task\Book1.xlsx",
                                         item_data_file=r"C:\Users\Syed Ashfaque Hussai\OneDrive\Desktop\CraftMyplate Machine Learning Task\item_data.csv")
    except Exception as e:
        print(f"Error loading and training model: {e}")
        return

    # Get predictions from terminal
    predictor.get_predictions_from_terminal()

if __name__ == "__main__":
    main()