"""
Practical Examples for Trademark Similarity API
Demonstrates common use cases and integration patterns
"""

import requests
import json
from typing import Dict, List
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

API_URL = "http://localhost:8000"


class TrademarkChecker:
    """Helper class for trademark similarity checking"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.session = requests.Session()
    
    def check_similarity(self, mark1: str, mark2: str, detailed: bool = False) -> Dict:
        """Check if two trademarks are similar"""
        response = self.session.post(
            f"{self.api_url}/similarity-check",
            json={
                "mark1": mark1,
                "mark2": mark2,
                "include_details": detailed
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def batch_check(self, pairs: List[tuple]) -> List[Dict]:
        """Check multiple trademark pairs"""
        formatted_pairs = [{"mark1": m1, "mark2": m2} for m1, m2 in pairs]
        response = self.session.post(
            f"{self.api_url}/batch-similarity",
            json={"pairs": formatted_pairs},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def print_result(self, mark1: str, mark2: str, result: Dict):
        """Pretty print a result"""
        # Color based on risk
        if result['risk_level'] == 'HIGH':
            risk_color = Fore.RED
        elif result['risk_level'] == 'MEDIUM':
            risk_color = Fore.YELLOW
        else:
            risk_color = Fore.GREEN
        
        print(f"\n{Fore.CYAN}Comparing:{Style.RESET_ALL}")
        print(f"  '{mark1}' vs '{mark2}'")
        print(f"\n{Fore.CYAN}Result:{Style.RESET_ALL}")
        print(f"  Similarity: {result['label_text']}")
        print(f"  Confidence: {result['probability']:.1%}")
        print(f"  {risk_color}Risk Level: {result['risk_level']}{Style.RESET_ALL}")
        print(f"  Recommendation: {result['recommendation']}")


def example_1_basic_check():
    """Example 1: Basic similarity check"""
    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}EXAMPLE 1: Basic Trademark Similarity Check{Style.RESET_ALL}")
    print("=" * 80)
    
    checker = TrademarkChecker()
    
    # Test case: Very similar marks
    mark1 = "SuperCoffee"
    mark2 = "Super Coffee"
    
    result = checker.check_similarity(mark1, mark2)
    checker.print_result(mark1, mark2, result)


def example_2_detailed_analysis():
    """Example 2: Detailed analysis with feature breakdown"""
    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}EXAMPLE 2: Detailed Feature Analysis{Style.RESET_ALL}")
    print("=" * 80)
    
    checker = TrademarkChecker()
    
    mark1 = "TechGenius Premium"
    mark2 = "PremiumTechGenius"
    
    result = checker.check_similarity(mark1, mark2, detailed=True)
    
    print(f"\n{Fore.CYAN}Trademarks:{Style.RESET_ALL}")
    print(f"  Mark 1: {mark1}")
    print(f"  Mark 2: {mark2}")
    
    print(f"\n{Fore.CYAN}Overall Assessment:{Style.RESET_ALL}")
    print(f"  Similarity: {result['probability']:.1%}")
    print(f"  Risk: {result['risk_level']}")
    
    if result.get('details'):
        details = result['details']
        
        print(f"\n{Fore.CYAN}Visual Similarity:{Style.RESET_ALL}")
        print(f"  Levenshtein Distance: {details['visual_features']['levenshtein_distance']}")
        print(f"  Jaro-Winkler Score: {details['visual_features']['jaro_winkler_similarity']:.3f}")
        
        print(f"\n{Fore.CYAN}Phonetic Analysis:{Style.RESET_ALL}")
        soundex = "✓ Match" if details['phonetic_features']['soundex_match'] else "✗ No match"
        metaphone = "✓ Match" if details['phonetic_features']['metaphone_match'] else "✗ No match"
        print(f"  Soundex: {soundex}")
        print(f"  Metaphone: {metaphone}")
        
        print(f"\n{Fore.CYAN}Semantic Similarity (Multilingual):{Style.RESET_ALL}")
        print(f"  English: {details['semantic_features']['similarity_en']:.3f}")
        print(f"  Hausa: {details['semantic_features']['similarity_ha']:.3f}")
        print(f"  Yoruba: {details['semantic_features']['similarity_yo']:.3f}")


def example_3_batch_processing():
    """Example 3: Batch processing multiple pairs"""
    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}EXAMPLE 3: Batch Processing{Style.RESET_ALL}")
    print("=" * 80)
    
    checker = TrademarkChecker()
    
    # Multiple trademark pairs to check
    pairs = [
        ("Nike", "Mike"),
        ("Apple", "Orange"),
        ("CoffeePlus", "PlusCoffee"),
        ("TechSmart", "SmartTech"),
        ("GlobalTrade", "TradeGlobal"),
        ("BrandNew", "NewBrand"),
    ]
    
    print(f"\nChecking {len(pairs)} trademark pairs...")
    
    results = checker.batch_check(pairs)
    
    print(f"\n{Fore.CYAN}Results Summary:{Style.RESET_ALL}\n")
    
    for (mark1, mark2), result in zip(pairs, results):
        # Format the pair
        pair_str = f"{mark1:20s} vs {mark2:20s}"
        
        # Color based on result
        if result['label'] == 1:
            status = f"{Fore.RED}✗ Similar{Style.RESET_ALL}"
        else:
            status = f"{Fore.GREEN}✓ Different{Style.RESET_ALL}"
        
        # Risk indicator
        if result['risk_level'] == 'HIGH':
            risk = f"{Fore.RED}HIGH{Style.RESET_ALL}"
        elif result['risk_level'] == 'MEDIUM':
            risk = f"{Fore.YELLOW}MED{Style.RESET_ALL}"
        else:
            risk = f"{Fore.GREEN}LOW{Style.RESET_ALL}"
        
        prob = f"{result['probability']:.1%}"
        
        print(f"  {pair_str} → {status} ({prob}, {risk} risk)")


def example_4_trademark_screening():
    """Example 4: Screen a new trademark against existing ones"""
    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}EXAMPLE 4: Trademark Portfolio Screening{Style.RESET_ALL}")
    print("=" * 80)
    
    checker = TrademarkChecker()
    
    # Proposed new trademark
    new_trademark = "TechMaster Pro"
    
    # Existing trademark portfolio
    existing_trademarks = [
        "TechMaster",
        "ProTech",
        "MasterTech",
        "TechExpert",
        "TechGenius",
        "DigitalPro",
        "SmartMaster",
    ]
    
    print(f"\n{Fore.CYAN}New Trademark:{Style.RESET_ALL} {new_trademark}")
    print(f"\n{Fore.CYAN}Checking against {len(existing_trademarks)} existing trademarks...{Style.RESET_ALL}\n")
    
    # Create pairs
    pairs = [(new_trademark, existing) for existing in existing_trademarks]
    results = checker.batch_check(pairs)
    
    # Analyze conflicts
    high_risk = []
    medium_risk = []
    low_risk = []
    
    for existing, result in zip(existing_trademarks, results):
        if result['risk_level'] == 'HIGH':
            high_risk.append((existing, result['probability']))
        elif result['risk_level'] == 'MEDIUM':
            medium_risk.append((existing, result['probability']))
        else:
            low_risk.append((existing, result['probability']))
    
    # Display results
    if high_risk:
        print(f"{Fore.RED}⚠ HIGH RISK CONFLICTS:{Style.RESET_ALL}")
        for mark, prob in high_risk:
            print(f"  • {mark:20s} ({prob:.1%} similarity)")
    
    if medium_risk:
        print(f"\n{Fore.YELLOW}⚠ MEDIUM RISK CONFLICTS:{Style.RESET_ALL}")
        for mark, prob in medium_risk:
            print(f"  • {mark:20s} ({prob:.1%} similarity)")
    
    if low_risk:
        print(f"\n{Fore.GREEN}✓ LOW RISK (SAFE):{Style.RESET_ALL}")
        for mark, prob in low_risk:
            print(f"  • {mark:20s} ({prob:.1%} similarity)")
    
    # Final recommendation
    print(f"\n{Fore.CYAN}Recommendation:{Style.RESET_ALL}")
    if high_risk:
        print(f"  {Fore.RED}❌ NOT RECOMMENDED{Style.RESET_ALL} - High risk conflicts detected")
        print(f"     Consider alternative trademark or modify '{new_trademark}'")
    elif medium_risk:
        print(f"  {Fore.YELLOW}⚠ CAUTION{Style.RESET_ALL} - Medium risk conflicts detected")
        print(f"     Legal review recommended before proceeding")
    else:
        print(f"  {Fore.GREEN}✓ PROCEED{Style.RESET_ALL} - No significant conflicts detected")
        print(f"     '{new_trademark}' appears available (subject to full legal review)")


def example_5_phonetic_similarity():
    """Example 5: Detecting phonetically similar marks"""
    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}EXAMPLE 5: Phonetic Similarity Detection{Style.RESET_ALL}")
    print("=" * 80)
    
    checker = TrademarkChecker()
    
    # Marks that sound similar
    phonetic_pairs = [
        ("Nike", "Mikey"),
        ("Coca-Cola", "KokaCola"),
        ("Microsoft", "MicroSoft"),
        ("Google", "Gooogle"),
    ]
    
    print(f"\n{Fore.CYAN}Testing phonetically similar trademarks:{Style.RESET_ALL}\n")
    
    for mark1, mark2 in phonetic_pairs:
        result = checker.check_similarity(mark1, mark2, detailed=True)
        
        print(f"\n'{mark1}' vs '{mark2}'")
        print(f"  Similarity: {result['probability']:.1%}")
        
        if result.get('details'):
            phonetic = result['details']['phonetic_features']
            soundex = "✓" if phonetic['soundex_match'] else "✗"
            metaphone = "✓" if phonetic['metaphone_match'] else "✗"
            
            print(f"  Soundex Match: {soundex}")
            print(f"  Metaphone Match: {metaphone}")
            print(f"  Risk: {result['risk_level']}")


def example_6_multilingual():
    """Example 6: Multilingual trademark analysis"""
    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}EXAMPLE 6: Multilingual Support{Style.RESET_ALL}")
    print("=" * 80)
    
    checker = TrademarkChecker()
    
    # Test with English, Hausa, and Yoruba terms
    multilingual_pairs = [
        ("SuperMarket", "BigStore"),  # English
        ("TechSmart", "SmartTech"),   # English
    ]
    
    print(f"\n{Fore.CYAN}Semantic similarity across languages:{Style.RESET_ALL}\n")
    
    for mark1, mark2 in multilingual_pairs:
        result = checker.check_similarity(mark1, mark2, detailed=True)
        
        print(f"\n'{mark1}' vs '{mark2}'")
        
        if result.get('details'):
            semantic = result['details']['semantic_features']
            print(f"  English:  {semantic['similarity_en']:.3f}")
            print(f"  Hausa:    {semantic['similarity_ha']:.3f}")
            print(f"  Yoruba:   {semantic['similarity_yo']:.3f}")
            print(f"  Overall:  {result['probability']:.1%} similarity")


def main():
    """Run all examples"""
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              TRADEMARK SIMILARITY API - PRACTICAL EXAMPLES                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)
    
    try:
        # Check if API is running
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code != 200:
            print(f"{Fore.RED}Error: API is not healthy. Please start the server first.{Style.RESET_ALL}")
            print(f"Run: python api_server.py")
            return
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}Error: Cannot connect to API at {API_URL}{Style.RESET_ALL}")
        print(f"Please start the server first:")
        print(f"  python api_server.py")
        return
    
    # Run examples
    examples = [
        example_1_basic_check,
        example_2_detailed_analysis,
        example_3_batch_processing,
        example_4_trademark_screening,
        example_5_phonetic_similarity,
        example_6_multilingual,
    ]
    
    for example in examples:
        try:
            example()
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error in {example.__name__}: {e}{Style.RESET_ALL}")
    
    print("\n" + "=" * 80)
    print(f"{Fore.GREEN}{Style.BRIGHT}✓ All examples completed!{Style.RESET_ALL}")
    print("=" * 80)
    print(f"\n{Fore.CYAN}For more information:{Style.RESET_ALL}")
    print(f"  • Interactive docs: http://localhost:8000/docs")
    print(f"  • Full documentation: API_DOCUMENTATION.md")
    print(f"  • Quick start: API_QUICKSTART.md")


if __name__ == "__main__":
    main()
