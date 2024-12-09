Building From Source
====================

Clone the Repository
---------------------
.. code-block:: bash

   git clone https://github.com/mlc-ai/web-llm.git
   cd web-llm

Install Dependencies
---------------------
.. code-block:: bash

   npm install

Build the Project
-----------------
.. code-block:: bash

   npm run build

Test Changes
------------

To test you changes, you can reuse any existing example or create a new example for your new functionality to test.

Then, to test the effects of your code change in an example, inside ``examples/<example>/package.json``, change from ``"@mlc-ai/web-llm": "^0.2.xx"`` to ``"@mlc-ai/web-llm": ../...`` to let it reference you local code.

.. code-block:: bash

   cd examples/<example>
   # Modify the package.json
   npm install
   npm start
