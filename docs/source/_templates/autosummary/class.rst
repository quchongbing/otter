{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   {% for item in attributes %}
   * ``{{ item }}``
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. _sphx_glr_backref_{{fullname}}:

   .. minigallery:: {{fullname}}
      :add-heading: Examples using ``{{ objname }}``
