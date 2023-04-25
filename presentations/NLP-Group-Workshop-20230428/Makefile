MMD_FILES := $(wildcard *.mmd)
PDF_FILES := $(MMD_FILES:.mmd=.pdf)

all: slide.pdf

slide.pdf: slide.md $(PDF_FILES)
	pandoc -t beamer -s -o slide.pdf slide.md

%.pdf: %.mmd
	mmdc -i $< -o $@ -f -b transparent

.PHONY: clean
clean:
	rm -f slide.pdf $(PDF_FILES)

.PHONY: watch
watch:
	@watchexec -e mmd,md 'date && make && printf "%s\n\n" Done'
