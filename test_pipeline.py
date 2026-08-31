from matcher import AlgorithmicMatcher


def run_smoke_tests():

    matcher = AlgorithmicMatcher("novo_vendor_master.csv", nrows=100_000)

    test_cases = [
        # Standard lookup with legal suffix
        {"name": "Infosys Ltd.",   "website": "infosys.com",   "country": "India"},
        # Lower-case / abbreviation variation
        {"name": "tata consultancy", "website": "tcs.com",     "country": "India"},
        # Partial name match
        {"name": "Cognizant Tech Solutions", "website": "cognizant.com", "country": "USA"},
        # Fake / unknown company
        {"name": "Nonexistent Corp X",       "website": "fakeurl123.com", "country": "Mars"},
        # Unicode accent test
        {"name": "Müller Industrietechnik",  "website": "",    "country": "Germany"},
        # Empty name, domain only
        {"name": "",                         "website": "andshayna.com", "country": ""},
        # Name only, no domain
        {"name": "Addinall Management",      "website": "",    "country": ""},
    ]

    print("=" * 60)
    print("  VENDOR MATCHER TEST SUITE")
    print("=" * 60, "\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: name='{test['name']}'  domain='{test['website']}'  country='{test['country']}'")

        matches = matcher.find_candidates(test, top_k=3)

        if not matches:
            print(f"  ➜ No matches found.\n")
            continue

        for rank, m in enumerate(matches, 1):
            print(
                f"  #{rank}  {m['name']}"
                f"  |  domain={m['website']}"
                f"  |  country={m['country_code']}"
                f"  |  composite={m['composite_score']}"
                f"  (name={m['name_score']}, domain={m['domain_score']})"
            )



if __name__ == "__main__":
    run_smoke_tests()