===========================
Read the Docs integration
===========================

Weas-widget relies on Jupyter widget state to render in static HTML. When
building on Read the Docs, make sure notebooks are executed and the widget
state is stored so the widgets appear in the final pages.

Configuration checklist
=======================

1. Install docs dependencies
----------------------------

Ensure your docs build environment includes ``nbsphinx`` and the Jupyter
runtime needed to execute notebooks (``ipykernel`` and ``ipython``). You can
either use a docs extra or a requirements file.

.. code-block:: yaml

   python:
     install:
       - method: pip
         path: .
         extra_requirements:
           - docs

2. Configure nbsphinx in ``conf.py``
-----------------------------------

Use the standard HTML widget manager and store widget state during execution.

.. code-block:: python

   # Always execute notebooks during docs build and persist widget state.
   nbsphinx_execute = "always"
   nbsphinx_execute_arguments = [
       "--ExecutePreprocessor.store_widget_state=True",
   ]
   # Use the standard HTML widget manager so widget state can render in static HTML.
   nbsphinx_widgets_path = (
       "https://unpkg.com/@jupyter-widgets/html-manager@^1.0.0/dist/embed-amd.js"
   )

Notes
=====

- Executing notebooks on Read the Docs increases build time; keep notebooks
  focused and cache-heavy cells to a minimum.
- If widgets still do not appear, confirm that your notebooks run cleanly in a
  fresh environment and that Read the Docs is using the same requirements.
