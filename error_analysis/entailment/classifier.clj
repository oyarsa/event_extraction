(ns classifier
  "Classification script to demonstrate how the entailment data is biased
  because of how it's constructed.

  When sentence1 is a substring of sentence2, the label is always 'ENTAILMENT'.
  For 'CONTRADICTION' and 'NEUTRAL', it's not as obvious."
  {:clj-kondo/ignore [:unresolved-namespace]}
  (:require [clojure.string :as str]
            [cheshire.core :as json]
            [clojure.java.io :as io]))

(defn levenshtein-distance
  "Levenshtein distance between two strings."
  [str1 str2]
  (let [m (count str1)
        n (count str2)
        dp (make-array Integer/TYPE (inc m) (inc n))]

    (dotimes [i (inc m)]
      (aset-int dp i 0 i))

    (dotimes [j (inc n)]
      (aset-int dp 0 j j))

    (dotimes [i m]
      (dotimes [j n]
        (let [cost (if (= (nth str1 i) (nth str2 j)) 0 1)]
          (aset-int dp (inc i) (inc j)
                    (min (+ (aget dp i j) cost)
                         (+ 1 (aget dp (inc i) j))
                         (+ 1 (aget dp i (inc j))))))))

    (aget dp m n)))

#_{:clj-kondo/ignore [:clojure-lsp/unused-public-var]}
(defn levenshtein-similarity
  "Similarity between two strings using the Levenshtein distance."
  [str1 str2]
  (let [len-sum (+ (count str1) (count str2))
        dist (levenshtein-distance str1 str2)]
    (float (- 1 (/ dist len-sum)))))

;; BAG OF WORDS
(defn set->map
  "Set and create a map from each element to an index."
  [s]
  (into {} (map vector s (range))))

(defn word-vector
  "Helper function for bag-of-words. Creates a vector of size |vocab|
  with the count of each word in words."
  [vocab words]
  (reduce (fn [acc word]
            (let [index (vocab word)]
              (update acc index inc)))
          (vec (repeat (count vocab) 0))
          words))

(defn split-words
  "Split a string on whitespace."
  [s]
  (clojure.string/split s #"\s+"))

(defn bag-of-words
  "Create a bag of words representation for two strings.
  The bag of words representation is a vector of size |vocab| where
  vocab is the union of unique of words between the strings."
  [vocab s]
  (word-vector vocab (split-words s)))

(defn make-vocab
  "Create a vocabulary from a collection of strings."
  [coll]
  (set->map (set (mapcat #(split-words %) coll))))
;; END OF BAG OF WORDS

(defn jaccard
  "Jaccard similarity between two vectors."
  [vector1 vector2]
  (let [sum-min (reduce + (map min vector1 vector2))
        sum-max (reduce + (map max vector1 vector2))]
    (float (/ sum-min sum-max))))

(defn norm
  "2-norm of a vector."
  [vector]
  (Math/sqrt (reduce + (map #(* % %) vector))))

(defn cosine
  "Cosine similarity between two vectors."
  [vector1 vector2]
  (let [dot-product (reduce + (map * vector1 vector2))
        norm1 (norm vector1)
        norm2 (norm vector2)]
    (float (/ dot-product (* norm1 norm2)))))

(defn jaccard-similarity
  "Jaccard similarity between two strings using the Generalised Jaccard index."
  [str1 str2]
  (let [vocab (make-vocab [str1 str2])]
    (jaccard (bag-of-words vocab str1)
             (bag-of-words vocab str2))))

#_{:clj-kondo/ignore [:clojure-lsp/unused-public-var]}
(defn cosine-similarity
  "Cosine similarity between two strings."
  [str1 str2]
  (let [vocab (make-vocab [str1 str2])]
    (cosine (bag-of-words vocab str1)
            (bag-of-words vocab str2))))

(def similarity-threshold 0.75)

(defn similar?
  "Returns true if the (sim-fn str1 str2) is greater
  than or equal to the similarity threshold."
  [str1 str2 sim-fn threshold]
  (>= (sim-fn str1 str2) threshold))

(defn symmetric-substring
  "Returns true if str1 is a substring of str2 or vice versa."
  [str1 str2]
  (or (str/includes? str1 str2) (str/includes? str2 str1)))

(defn classify
  "Classify two strings as entailment, contradiction or neutral."
  ([str1 str2 sim-fn threshold]
   (cond
     (symmetric-substring str1 str2) "ENTAILMENT"
     (similar? str1 str2 sim-fn threshold) "CONTRADICTION"
     :else "NEUTRAL")))

(defn classify*
  "Classify a collection of maps with :sentence1 and :sentence2 keys."
  [coll sim-fn threshold]
  (map #(classify (:sentence1 %) (:sentence2 %) sim-fn threshold)
       coll))

(defn calc-accuracy
  "Calculate the accuracy of the classifier on a collection of maps with
  :sentence1, :sentence2 and :label keys."
  [coll sim-fn threshold]
  (let [gold (map :label coll)
        pred (classify* coll sim-fn threshold)
        correct (filter #(apply = %) (map vector gold pred))]
    (float (/ (count correct) (count coll)))))

(defn get-sim-fn
  "Get the similarity function from a string."
  [s]
  (case s
    "jaccard" jaccard-similarity
    "cosine" cosine-similarity
    "levenshtein" levenshtein-similarity
    (throw (Exception. (str "Unknown similarity function: " s)))))

(def opt-spec
  {:coerce {:max-samples :long
            :threshold :double}
   :alias {:m :max-samples}
   :exec-args {:similarity "jaccard"
               :threshold similarity-threshold}
   :args->opts [:data-file]})

(let [opts (babashka.cli/parse-opts *command-line-args* opt-spec)
      data-file (:data-file opts)
      sim-fn (get-sim-fn (:similarity opts))
      max-samples (:max-samples opts)
      threshold (:threshold opts)]
  (when (empty? data-file)
    (println "Usage: bb levenshtein.clj <data-file>")
    (System/exit 1))

  (let [data (json/parse-stream (io/reader data-file) true)
        n (or max-samples (count data))
        score (calc-accuracy (take n data) sim-fn threshold)]
    (println "Accuracy" (format "%.2f%%" (* 100 score)))))

(let [str1 "hey there man hey"
      str2 "hey there girl there there"
      vocab (make-vocab [str1 str2])
      bow1 (bag-of-words vocab str1)
      bow2 (bag-of-words vocab str2)]
  (println bow1 "=>" str1)
  (println bow2 "=>" str2))
