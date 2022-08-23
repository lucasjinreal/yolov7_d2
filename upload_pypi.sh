##
## Copyright (c) 2020 JinTian.
##
## This file is part of alfred
## (see http://jinfagang.github.io).
##
## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing,
## software distributed under the License is distributed on an
## "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
## KIND, either express or implied.  See the License for the
## specific language governing permissions and limitations
## under the License.
##
# check setup is correct or not
python3 setup.py check

sudo rm -r build/
sudo rm -r dist/

# pypi interface are not valid any longer
# python3 setup.py sdist
# python3 setup.py sdist upload -r pypi

# using twine instead
python3 setup.py sdist
twine upload dist/*

