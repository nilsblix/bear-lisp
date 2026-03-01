; compose function
(define o (f g) (lambda (x) (f (g x))))

(define map (f xs)
  (pair (f (car xs))
        (let ((rest (cdr xs)))
          (if (eq rest ())
            ()
            (map f rest)))))
