; compose function
(define o (f g) (lambda (x) (f (g x))))
(val caar (o car car))
(val cadr (o car cdr))
(val caddr (o cadr cdr))
(val cadar (o car (o cdr car)))
(val caddar (o car (o cdr (o cdr car))))

(val cons pair)

(val newline (itoc 10))
(val space (itoc 32))

(define getline ()
  (let* ((ic (getchar))
         (c (itoc ic)))
    (if (or (eq c newline) (eq ic ~1))
      empty-symbol
      (cat c (getline)))))

(define null? (xs) (eq xs '()))

(define length (xs)
  (if (null? xs)
    0
    (+ 1 (length (cdr xs)))))

(define take (n xs)
  (if (or (< n 1) (null? xs))
    '()
    (cons (car xs) (take (- n 1) (cdr xs)))))

(define drop (n xs)
  (if (or (< n 1) (null? xs))
    xs
    (drop (- n 1) (cdr xs))))

(define merge (xs ys)
  (if (null? xs)
    ys
    (if (null? ys)
      xs
      (if (< (car xs) (car ys))
        (cons (car xs) (merge (cdr xs) ys))
        (cons (car ys) (merge xs (cdr ys)))))))

(define mergesort (ls)
  (if (null? ls)
    ls
    (if (null? (cdr ls))
      ls
      (let* ((size (length ls))
             (half (/ size 2))
             (first (take half ls))
             (second (drop half ls)))
        (merge (mergesort first) (mergesort second))))))

(define map (f xs)
  (pair (f (car xs))
        (let ((rest (cdr xs)))
          (if (null? rest)
            ()
            (map f rest)))))

(define mem (x xs)
  (if (null? xs)
    #f
    (or (eq x (car xs)) (mem x (cdr xs)))))

(define find (pred xs)
  (if (null? xs)
    'not_found
    (let ((elem (car xs)))
      (if (pred elem)
        elem
        (find pred (cdr xs))))))

(define filter (f xs)
  (if (null? xs)
    '()
    (let ((elem (car xs)))
      (if (f elem)
        (cons elem (filter f (cdr xs)))
        (filter f (cdr xs))))))
