Using Agent
===========

``WeasToolkit`` exposes LangChain-compatible tools that let an agent inspect and edit a ``WeasWidget``.
One can use the `langchain-widget` package to create a chat agent with the toolkit. First, install the required packages:

.. code-block:: bash

   pip install langchain-widget[openai]


Then, you can create a toolkit instance by passing in a ``WeasWidget`` instance.

.. important::

   The ``WeasToolkit`` requires an LLM that supports function calling, such as GPT.
   Make sure to set up your OpenAI API key in the environment variable ``OPENAI_API_KEY``.
   Check the `LangChain OpenAI documentation <https://langchain-widget.readthedocs.io/en/latest/quick_start.html>`_ for more details.

Example
-------

Run the following code to create a LangChain chat widget integrated with `weas_widget`. And try to ask:

- “Load a Si diamond conventional cell and repeat 2x2x2”
- “Select atom 0 and atom 1.”
- “Replace the selected atoms with Ge”
- "Summarize the structure"


.. code-block:: python

   from weas_widget import WeasWidget, WeasToolkit
   from langchain_openai import ChatOpenAI
   from langchain_widget import LangChainWidget
   from langchain_openai import ChatOpenAI
   from dotenv import load_dotenv
   import ipywidgets as ipw

   load_dotenv()

   viewer = WeasWidget()

   chat_model = ChatOpenAI(model="gpt-4o")
   chat = LangChainWidget(
      chat_model=chat_model,
      tools=WeasToolkit(viewer=viewer),
      title="WEAS Agent Chat",
      system_prompt=(
         "You are a scientific assistant. "
         "Use the available tools to inspect and manipulate the 3D structure."
      ),
      sidebar_open=False,
   )
   ipw.HBox([viewer, chat])

.. image:: ../_static/images/langchain-agent.png
   :alt: Agent Tools Example
   :align: center
   :width: 100%

Extending the toolkit
---------------------

You can add your own tools in two ways:

1) Pass tools directly:

.. code-block:: python

   toolkit = WeasToolkit(viewer, extra_tools=[my_tool])
   tools = toolkit.tools

2) Register entry points under ``weas_widget.tools`` that return a tool or a tool factory.
