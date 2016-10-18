#!/usr/bin/env python3
import sys, re, os
rttype, basename = sys.argv[1:]

if rttype in ("nvvm", "spir"):
    # we need to patch
    result = []
    filename = basename+"."+rttype
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                if rttype=="nvvm":
                    # patch to opaque identity functions
                    m = re.match('^declare (.*) @(magic_.*_id)\((.*)\)\n$', line)
                    if m is not None:
                        ty1, fname, ty2 = m.groups()
                        assert ty1 == ty2, "Argument and return types of magic IDs must match"
                        print("Patching magic ID {0}".format(fname))
                        # emit definition instead
                        result.append('define {0} @{1}({0} %name) {{\n'.format(ty1, fname))
                        result.append('  ret {0} %name\n'.format(ty1))
                        result.append('}\n')
                    # it's a normal line, keep it
                    else:
                        result.append(line)
                else: # spir
                    # patch to opaque identity functions
                    m = re.match('^declare cc75 (.*) @(magic_.*_id)\((.*)\)\n$', line)
                    if m is not None:
                        ty1, fname, ty2 = m.groups()
                        assert ty1 == ty2, "Argument and return types of magic IDs must match"
                        print("Patching magic ID {0}".format(fname))
                        # emit definition instead
                        result.append('define cc75 {0} @{1}({0} %name) {{\n'.format(ty1, fname))
                        result.append('  ret {0} %name\n'.format(ty1))
                        result.append('}\n')
                    # get rid of attributes
                    elif 'attributes #' in line:
                        print("Removing attribute declarations")
                        # ignore this line
                        pass
                    # emit correct datalayout for llvm 3.6
                    elif 'target datalayout = "e-i32' in line:
                        result.append('target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"\n')
                    elif 'target datalayout = "e-i64' in line:
                        result.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"\n')
                    # it's a normal line, keep it but apply substitutions
                    else:
                        # get rid of attributes at end of line
                        line = re.sub('#[0-9]+', '', line)
                        # add metadata for llvm 3.6
                        line = re.sub(r'\A!([0-9]+) = !', r'!\1 = metadata !', line)
                        line = re.sub(r', !"kernel",', r', metadata !"kernel",', line)
                        line = re.sub(r', !"texture",', r', metadata !"texture",', line)
                        result.append(line)
        # we have the patched thing, write it
        with open(filename, "w") as f:
            for line in result:
                f.write(line)

if rttype in ("cuda", "opencl"):
    # we need to patch
    result = []
    if rttype == "cuda":
        filename = basename+"."+"cu"
    else:
        filename = basename+"."+"cl"
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                # patch to opaque identity functions
                m = re.match('^(.*) = (magic_.*_id)\((.*)\);\n$', line)
                if m is not None:
                    lhs, fname, arg = m.groups()
                    print("Patching magic ID {0}".format(fname))
                    # emit definition instead
                    result.append('{0} = {1};\n'.format(lhs, arg))
                else:
                    result.append(line)
        # we have the patched thing, write it
        with open(filename, "w") as f:
            for line in result:
                f.write(line)

# another pass to add the ldg, minmax and consorts to the nvvm file
nvvm_defs = {
}

if rttype == "nvvm":
    result = []
    filename = basename+".nvvm"
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                matched = False

                for (func, code) in nvvm_defs.iteritems():
                    m = re.match('^declare (.*) (@' + func + ')\((.*)\)\n$', line)
                    if m is not None:
                        result.append(code)
                        matched = True
                        break

                if not matched:
                    result.append(line)

        with open(filename, "w") as f:
            for line in result:
                f.write(line)

