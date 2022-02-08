
import os
import glob
import json
import textwrap

def rip():

    targets = [
        'book/example/*.ipynb',
    ]

    generator_dest = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "larch", "examples", "generated",
    ))
    os.makedirs(generator_dest, exist_ok=True)
    print(f"{generator_dest=}")

    with open(os.path.join(generator_dest, ".gitignore"), 'wt') as f:
        f.write(".gitignore\n*.py")

    import_stmts = []
    assign_stmts = []

    for target in targets:
        for examplefile in glob.glob(target):
            print(f"{examplefile=}")
            example = os.path.basename(examplefile).replace(".ipynb", "")
            example = example.replace("-","_")
            print(f"  {example=}")
            with open(os.path.join(generator_dest, f"_{example}.py"), 'wt') as f:
                f.write("def example(extract='m', estimate=False):\n")
                with open(examplefile) as s:
                    sourcefilecontent = s.read()
                y = json.loads(sourcefilecontent)
                source1 = []
                source2 = []
                trip = False
                for i in y['cells']:
                    if (
                            i['cell_type'] == 'code'
                            and not i['metadata'].get('doc_only', False)
                            and not i['metadata'].get('remove_cell', False)
                            and 'remove_cell' not in i.get('metadata', {}).get('tags', [])
                    ):
                        s = "".join(i['source'])
                        if ".maximize_loglike(" in s:
                            trip = True
                        if trip:
                            source2.append(s)
                        else:
                            source1.append(s)
                sourcecode1 = "\n\n".join(source1)
                sourcecode2 = "\n\n".join(source2)
                f.write(textwrap.indent(sourcecode1, "    "))
                mid = """
                if not estimate:
                    if isinstance(extract, str):
                        return locals()[extract]
                    else:
                        _locals = locals()
                        return [_locals.get(i) for i in extract]
                """
                f.write(textwrap.indent(textwrap.dedent(mid), "    "))
                f.write(textwrap.indent(sourcecode2, "    "))
                post = """
                if isinstance(extract, str):
                    return locals()[extract]
                else:
                    _locals = locals()
                    return [_locals.get(i) for i in extract]
                """
                f.write(textwrap.indent(textwrap.dedent(post), "    "))

            ex_n = example.split("_")[0]
            try:
                ex_n = int(ex_n)
            except ValueError:
                pass
            import_stmts.append(f"from ._{example} import example as _{example}")
            assign_stmts.append(f"ex[{ex_n!r}] = _{example}")

    with open(os.path.join(generator_dest, "__init__.py"), 'wt') as gen:
        gen.write(textwrap.dedent("""
        # Example code extracted automatically from Larch documentation.
        # Do not edit these files manually.
        """))
        for s in import_stmts:
            gen.write(s)
            gen.write("\n")
        gen.write("\nex = {}\n")
        for s in assign_stmts:
            gen.write(s)
            gen.write("\n")


if __name__ == '__main__':
    rip()
