#!/usr/bin/env bb

(require '[cheshire.core :as json])

(defn read-json-file [file-path]
  (json/parse-string (slurp file-path) true))

(defn create-table [labels]
  (let [zero-row (zipmap labels (repeat 0))]
    (zipmap labels (repeat zero-row))))

(defn create-confusion-table [data]
  (let [gold-pred-pairs (map (juxt :gold :prediction) data)
        labels (distinct (flatten gold-pred-pairs))
        table (create-table labels)]
    (reduce (fn [acc [gold prediction]]
              (update-in acc [gold prediction] inc))
            table
            gold-pred-pairs)))

(defn print-confusion-table [table]
  (let [labels (keys table)
        max-label-length (apply max (map count labels))
        padding-format (str "%" (+ 2 max-label-length) "s")
        format-entry (fn [entry] (format padding-format entry))]
    (print (format-entry "G\\P"))
    (doseq [label labels]
      (print (format-entry label)))
    (println)
    (doseq [row-label labels]
      (print (format-entry row-label))
      (doseq [col-label labels]
        (print (format-entry (get-in table [row-label col-label]))))
      (println))))

(defn calculate-accuracy [true-labels prediction-labels]
  (let [zipped (map vector true-labels prediction-labels)
        label-counts (frequencies true-labels)
        matched (filter #(= (first %) (second %)) zipped)
        matched-counts (frequencies (map first matched))]
    (into {} (map (fn [[label count]]
                    [label (float (/ (matched-counts label 0) count))])
                  label-counts))))

(defn print-accuracy [accuracy]
  (let [max-label-length (apply max (map count (distinct (keys accuracy))))
        padding-format (str "%" (+ 2 max-label-length) "s")]
    (doseq [[label value] accuracy]
      (println (format (str padding-format ": %6.2f%%") label (* 100 value))))))

(defn -main [& [file-path]]
  (let [data (read-json-file file-path)
        table (create-confusion-table data)
        accuracy (calculate-accuracy (map :gold data) (map :prediction data))]
    (println "Confusion table")
    (print-confusion-table table)
    (println)
    (println "Class accuracy")
    (print-accuracy accuracy)))

(when (= *file* (System/getProperty "babashka.file"))
  (apply -main *command-line-args*))

(comment
  (def data (read-json-file "eval_output.json"))
  (def dd (take 5 data))
  dd
  (create-confusion-table dd)
  (print-confusion-table (create-confusion-table dd))

  (create-table ["A" "B" "C"])
  2
  (calculate-accuracy [:a :a :b :b :c] [:a :b :b :b :b])
  (print-accuracy
   (calculate-accuracy ["a" "a" "b" "b" "cc"] ["a" "b" "b" "b" "b"]))

  :rcf)