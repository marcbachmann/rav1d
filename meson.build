# Copyright © 2018-2022, VideoLAN and dav1d authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Runs the conformance vectors in tests/dav1d-test-data against a cargo-built
# dav1d binary; see .github/workflows/test.sh. Nothing is compiled here.
project('rav1d-tests', meson_version: '>= 0.49.0')

if get_option('test_rust_path') == ''
    error('pass -Dtest_rust_path=<path to the cargo-built dav1d binary>')
endif
dav1d = files(get_option('test_rust_path'))

# Without a seek_stress binary its tests are registered against dav1d but only
# run when the testdata_seek-stress suite is asked for.
if get_option('seek_stress_test_rust_path') != ''
    seek_stress = files(get_option('seek_stress_test_rust_path'))
else
    seek_stress = dav1d
endif

dav1d_bitdepths = get_option('bitdepths')

subdir('tests/dav1d-test-data')
