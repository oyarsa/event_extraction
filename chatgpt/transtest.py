from google_trans_new import google_translator

t = google_translator()
print(t.translate("안녕하세요.", lang_tgt="en"))
print(t.translate("oi", lang_src="pt", lang_tgt="en"))
