from spell_check_module import loadSpellCheck

spellChecker = loadSpellCheck('model/typo-correction', 'model/data.json')

print(spellChecker.fix_sentence('ak maw cuti bsk'))
print(spellChecker.fix_sentence('cuti bsk sampe 20 jun'))