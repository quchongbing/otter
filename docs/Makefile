# Local Sphinx entry points. Generated files are deliberately untracked.

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean strict serve

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf source/_autosummary source/gen_examples source/gen_modules \
		source/benchmarks/gen_benchmarks
	rm -f source/sg_execution_times.rst

# CI/release-quality build: start from fresh generated sources and turn every
# Sphinx warning into a failure while still reporting all warnings.
strict: clean
	@$(SPHINXBUILD) -E -a -W --keep-going -b html \
		"$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Serve only on localhost; this is a private development preview, not deploy.
serve:
	@python -m http.server --bind 127.0.0.1 \
		--directory "$(BUILDDIR)/html" 8000

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
