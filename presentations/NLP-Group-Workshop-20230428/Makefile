slide.pdf: slide.md rl.pdf
	pandoc -t beamer -o slide.pdf slide.md

rl.pdf: rl.mmd
	mmdc -i rl.mmd -o rl.pdf -f -b transparent

.PHONY: clean
clean:
	rm -f slide.pdf rl.pdf
