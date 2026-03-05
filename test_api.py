"""
Comprehensive API Testing Script
Tests all endpoints and demonstrates trademark similarity detection
"""

import requests
import json
import time
from typing import Dict, Any
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

API_BASE_URL = "http://localhost:8000"

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print("=" * 80)

def print_success(text: str):
    """Print success message"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text: str):
    """Print error message"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text: str):
    """Print info message"""
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")

def print_result(label: str, value: Any, color=Fore.WHITE):
    """Print a labeled result"""
    print(f"  {color}{label}: {value}{Style.RESET_ALL}")


class APITester:
    """Test the Trademark Similarity API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        print_header("TEST 1: Health Check")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            print_result("Status", data['status'], 
                        Fore.GREEN if data['status'] == 'healthy' else Fore.RED)
            print_result("Models Loaded", data['models_loaded'],
                        Fore.GREEN if data['models_loaded'] else Fore.RED)
            print_result("Timestamp", data['timestamp'])
            
            if data.get('model_info'):
                print("\n  Model Components:")
                for key, val in data['model_info'].items():
                    status = "✓" if val else "✗"
                    color = Fore.GREEN if val else Fore.RED
                    print(f"    {color}{status} {key}: {val}{Style.RESET_ALL}")
            
            if data['models_loaded']:
                print_success("Health check passed")
                return True
            else:
                print_error("Models not loaded")
                return False
                
        except Exception as e:
            print_error(f"Health check failed: {e}")
            return False
    
    def test_root(self):
        """Test root endpoint"""
        print_header("TEST 2: Root Endpoint")
        
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            print_result("API Name", data['name'])
            print_result("Version", data['version'])
            print_result("Model", data['model'])
            print_result("Languages", ", ".join(data['languages']))
            
            print("\n  Available Endpoints:")
            for name, path in data['endpoints'].items():
                print(f"    {Fore.CYAN}{name:20s}{Style.RESET_ALL} → {path}")
            
            print_success("Root endpoint accessible")
            
        except Exception as e:
            print_error(f"Root endpoint failed: {e}")
    
    def test_similarity_simple(self):
        """Test simple similarity check"""
        print_header("TEST 3: Simple Similarity Check")
        
        test_cases = [
            ("SuperCoffee", "Super Coffee", "Identical (spacing)"),
            ("TechSmart", "SmartTech", "Word rearrangement"),
            ("Nike", "Mike", "One letter difference"),
            ("Apple", "Orange", "Completely different"),
        ]
        
        for mark1, mark2, description in test_cases:
            print(f"\n{Fore.YELLOW}Case: {description}{Style.RESET_ALL}")
            print(f"  Mark 1: {mark1}")
            print(f"  Mark 2: {mark2}")
            
            try:
                response = self.session.post(
                    f"{self.base_url}/similarity-check",
                    json={"mark1": mark1, "mark2": mark2},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                
                # Color based on result
                if result['label'] == 1:
                    label_color = Fore.RED
                    prob_color = Fore.RED if result['probability'] > 0.7 else Fore.YELLOW
                else:
                    label_color = Fore.GREEN
                    prob_color = Fore.GREEN
                
                print_result("Result", result['label_text'], label_color)
                print_result("Probability", f"{result['probability']:.1%}", prob_color)
                print_result("Risk Level", result['risk_level'], 
                           Fore.RED if result['risk_level'] == 'HIGH' else 
                           Fore.YELLOW if result['risk_level'] == 'MEDIUM' else Fore.GREEN)
                print_result("Recommendation", result['recommendation'][:60] + "...")
                
                print_success(f"Prediction completed in {response.elapsed.total_seconds():.3f}s")
                
            except Exception as e:
                print_error(f"Failed: {e}")
    
    def test_similarity_detailed(self):
        """Test similarity with detailed breakdown"""
        print_header("TEST 4: Detailed Similarity Analysis")
        
        mark1 = "TechGenius Premium"
        mark2 = "PremiumTechGenius"
        
        print(f"Mark 1: {mark1}")
        print(f"Mark 2: {mark2}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/similarity-check",
                json={
                    "mark1": mark1,
                    "mark2": mark2,
                    "include_details": True
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            print_result("Result", result['label_text'], 
                        Fore.RED if result['label'] == 1 else Fore.GREEN)
            print_result("Probability", f"{result['probability']:.1%}")
            print_result("Risk", result['risk_level'])
            
            if result.get('details'):
                details = result['details']
                
                print(f"\n{Fore.CYAN}Visual Features:{Style.RESET_ALL}")
                for key, val in details['visual_features'].items():
                    print(f"  {key}: {val}")
                
                print(f"\n{Fore.CYAN}Phonetic Features:{Style.RESET_ALL}")
                for key, val in details['phonetic_features'].items():
                    status = "✓" if val else "✗"
                    color = Fore.GREEN if val else Fore.RED
                    print(f"  {color}{status} {key}{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}Semantic Features:{Style.RESET_ALL}")
                for key, val in details['semantic_features'].items():
                    print(f"  {key}: {val:.4f}")
                
                print(f"\n{Fore.CYAN}Length Features:{Style.RESET_ALL}")
                for key, val in details['length_features'].items():
                    print(f"  {key}: {val}")
                
                print(f"\n{Fore.CYAN}Model Info:{Style.RESET_ALL}")
                print(f"  CNN Embedding Size: {details['cnn_embedding_size']}")
                print(f"  Total Features: {details['total_features']}")
            
            print_success("Detailed analysis completed")
            
        except Exception as e:
            print_error(f"Failed: {e}")
    
    def test_batch(self):
        """Test batch processing"""
        print_header("TEST 5: Batch Processing")
        
        pairs = [
            {"mark1": "BrandA", "mark2": "BrandB"},
            {"mark1": "CoffeePlus", "mark2": "PlusCoffee"},
            {"mark1": "TechSmart", "mark2": "TechGenius"},
            {"mark1": "SuperMarket", "mark2": "MarketPro"},
            {"mark1": "GlobalTrade", "mark2": "TradeGlobal"},
        ]
        
        print(f"Testing {len(pairs)} trademark pairs...")
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/batch-similarity",
                json={"pairs": pairs},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            
            elapsed = time.time() - start_time
            
            print(f"\n{Fore.CYAN}Results:{Style.RESET_ALL}")
            for i, (pair, result) in enumerate(zip(pairs, results), 1):
                status_color = Fore.RED if result['label'] == 1 else Fore.GREEN
                print(f"\n  Pair {i}: {pair['mark1']} vs {pair['mark2']}")
                print(f"    {status_color}→ {result['label_text']}{Style.RESET_ALL} "
                      f"({result['probability']:.1%} confidence, {result['risk_level']} risk)")
            
            print(f"\n{Fore.CYAN}Performance:{Style.RESET_ALL}")
            print(f"  Total time: {elapsed:.3f}s")
            print(f"  Avg per pair: {elapsed/len(pairs):.3f}s")
            
            print_success("Batch processing completed")
            
        except Exception as e:
            print_error(f"Failed: {e}")
    
    def test_features_extraction(self):
        """Test feature extraction endpoint"""
        print_header("TEST 6: Feature Extraction")
        
        mark1 = "SuperCoffee"
        mark2 = "Super Coffee"
        
        print(f"Mark 1: {mark1}")
        print(f"Mark 2: {mark2}")
        
        try:
            response = self.session.get(
                f"{self.base_url}/features",
                params={"mark1": mark1, "mark2": mark2},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            features = data['features']
            
            print(f"\n{Fore.CYAN}Extracted Features:{Style.RESET_ALL}")
            for key, val in features.items():
                if isinstance(val, bool):
                    status = "✓" if val else "✗"
                    color = Fore.GREEN if val else Fore.RED
                    print(f"  {color}{status} {key}{Style.RESET_ALL}")
                elif isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")
            
            print_success("Feature extraction completed")
            
        except Exception as e:
            print_error(f"Failed: {e}")
    
    def test_stats(self):
        """Test stats endpoint"""
        print_header("TEST 7: API Statistics")
        
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            print(f"{Fore.CYAN}Model Architecture:{Style.RESET_ALL}")
            for key, val in data['models'].items():
                print(f"  {key}: {val}")
            
            print(f"\n{Fore.CYAN}Performance Metrics:{Style.RESET_ALL}")
            for key, val in data['performance'].items():
                print(f"  {key}: {val:.4f}")
            
            print(f"\n{Fore.CYAN}Configuration:{Style.RESET_ALL}")
            for key, val in data['config'].items():
                print(f"  {key}: {val}")
            
            print(f"\n{Fore.CYAN}Limits:{Style.RESET_ALL}")
            for key, val in data['limits'].items():
                print(f"  {key}: {val}")
            
            print_success("Statistics retrieved")
            
        except Exception as e:
            print_error(f"Failed: {e}")
    
    def test_get_method(self):
        """Test GET method for similarity"""
        print_header("TEST 8: GET Method (Query Parameters)")
        
        params = {
            "mark1": "Nike",
            "mark2": "Mike",
            "include_details": "false"
        }
        
        url = f"{self.base_url}/similarity-check"
        print(f"URL: {url}")
        print(f"Params: {params}")
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            print_result("Result", result['label_text'])
            print_result("Probability", f"{result['probability']:.1%}")
            print_result("Risk", result['risk_level'])
            
            print_success("GET method works")
            
        except Exception as e:
            print_error(f"Failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}")
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║         TRADEMARK SIMILARITY ENGINE - COMPREHENSIVE API TEST              ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print(Style.RESET_ALL)
        
        print_info(f"Testing API at: {self.base_url}")
        print_info("Waiting for server to be ready...")
        time.sleep(1)
        
        # Test health first
        if not self.test_health():
            print_error("\n❌ API not healthy. Please ensure the server is running:")
            print("   python api_server.py")
            return
        
        # Run all tests
        tests = [
            self.test_root,
            self.test_similarity_simple,
            self.test_similarity_detailed,
            self.test_batch,
            self.test_features_extraction,
            self.test_stats,
            self.test_get_method
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print_error(f"Test {test.__name__} failed with unexpected error: {e}")
        
        # Final summary
        print_header("TEST SUMMARY")
        print_success("All tests completed!")
        print_info("\n📚 For more information, visit: http://localhost:8000/docs")
        print_info("📖 Documentation: API_DOCUMENTATION.md")


def main():
    """Main function"""
    tester = APITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
