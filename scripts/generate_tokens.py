import secrets
import string

def generate_secure_token(length=32):
    """Generate cryptographically secure token"""
    alphabet = string.ascii_letters + string.digits + '-_'
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_multiple_tokens(count=3, length=32):
    """Generate multiple tokens for different users/apps"""
    tokens = []
    for i in range(count):
        token = generate_secure_token(length)
        tokens.append(token)
        print(f"Token {i+1}: {token}")
    
    # Format untuk Modal secret
    tokens_string = ','.join(tokens)
    print(f"\n📋 Copy this for Modal secret:")
    print(f"VALID_TOKENS=\"{tokens_string}\"")
    
    return tokens

if __name__ == "__main__":
    print("🔐 Generating secure Bearer tokens...\n")
    
    # Generate 3 tokens (bisa untuk different clients/apps)
    tokens = generate_multiple_tokens(3, 32)
    
    print(f"\n💡 Usage:")
    print(f"1. Update Modal secret: modal secret create upscaler-auth --force VALID_TOKENS=\"{','.join(tokens)}\"")
    print(f"2. Use any of these tokens in Authorization header: Bearer <token>")
    print(f"3. Keep tokens secure - treat like passwords!")