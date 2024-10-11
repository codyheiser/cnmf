# define directories
SRC_DIR = cNMF
DOCS_DIR = docs

# define tools
LINTER = flake8
FORMATTER = black
DOC_TOOL = pdoc3

# lint the code
lint:
	$(LINTER) $(SRC_DIR)

# format the code
format:
	$(FORMATTER) $(SRC_DIR)

# generate documentation
doc:
	$(DOC_TOOL) --html $(SRC_DIR) -o $(DOCS_DIR) --force
	mv $(DOCS_DIR)/cNMF/* $(DOCS_DIR)
	rm -r $(DOCS_DIR)/cNMF/

# run all tasks
all: lint format docs

.PHONY: lint format doc all