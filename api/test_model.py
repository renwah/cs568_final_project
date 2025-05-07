import spacy
import textdescriptives
import joblib
import os
import sys

def test_spacy():
    """Test spaCy and textdescriptives setup"""
    print("\nTesting spaCy and textdescriptives...")
    try:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textdescriptives/all")
        
        # Test processing a simple text
        text = "This is a test sentence."
        doc = nlp(text)
        
        # Test that we can access textdescriptives attributes
        readability = doc._.readability
        print("✓ spaCy and textdescriptives working correctly")
        return True
    except Exception as e:
        print(f"✗ Error testing spaCy: {str(e)}")
        return False

def test_model_files():
    """Check if model files exist"""
    print("\nChecking for model files...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    required_files = ['xgb_model.joblib', 'label_encoder.joblib', 'feature_columns.joblib']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing model files: {', '.join(missing_files)}")
        print(f"Please copy these files from the XGBoost notebook output to the {model_path} directory")
        return False
    
    print("✓ All model files present")
    return True

def main():
    print("Starting model setup tests...")
    
    # Test spaCy setup
    spacy_ok = test_spacy()
    
    # Test model files
    model_files_ok = test_model_files()
    
    # Summary
    print("\nTest Summary:")
    print(f"spaCy and textdescriptives: {'✓' if spacy_ok else '✗'}")
    print(f"Model files: {'✓' if model_files_ok else '✗'}")
    
    if not (spacy_ok and model_files_ok):
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)
    
    print("\n✅ All tests passed! Ready to proceed with server setup.")

if __name__ == "__main__":
    main() 