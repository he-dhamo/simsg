# Copyright 2020 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

num=0;
res_dir='./MyClevr';
for j in */; do 
find $(pwd)/$j"images" -iname "*_target.png" | sort -u | while read p; do
  cp $p $res_dir"/target/images/$num.png";
  q=$(pwd)/$j"scenes/${p##*/}"
  q="${q%%.*}.json"
  cp $q $res_dir"/target/scenes/$num.json";
  ((num++));
  echo $p;
  echo $q;
  echo $num;
done
num=`expr $(ls -1 "$res_dir/target/images/" | wc -l)`;
done
