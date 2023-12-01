import re

# Phrase de test
text = "4, 4,000 4.2%"

# Expression régulière
pattern = (
            # r"\w+(?:[-./']\w+)*|\w+(?:[-./']\w+(?:[-./']\w+)*)|"# This will match words like "well-known"
            # r'(?:[A-Z]\.)+|'    # This will match words like "U.S.A" or "U.S.A."
            # r'\"[^\"]+\"|'      # This will match words inside of quotes
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+%?)?'
            
        )
# Trouver les correspondances dans la phrase
matches = re.findall(pattern, text)

# Afficher les mots correspondants
for match in matches:
    print(match)