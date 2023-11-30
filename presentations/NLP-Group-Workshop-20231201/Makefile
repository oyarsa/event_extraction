all: slide.pdf

slide.pdf: slide.md
	pandoc -t beamer -s -o slide.pdf slide.md

.PHONY: clean
clean:
	rm -f slide.pdf

.PHONY: watch
watch:
	@watchexec -e mmd,md 'date && make && printf "%s\n\n" Done'
