import bisect

class ConvexPolygonMaxRectangle:
    def __init__(self, vertices):
        """
        vertices: list of (x, y) in CCW order
        """
        self.vertices = vertices
        self.n = len(vertices)
        self._prepare_chains()

    # -----------------------------
    # Step 1: split into upper/lower chains
    # -----------------------------
    def _prepare_chains(self):
        pts = self.vertices

        # find leftmost and rightmost points
        left = min(range(self.n), key=lambda i: pts[i][0])
        right = max(range(self.n), key=lambda i: pts[i][0])

        # build lower chain (left → right)
        i = left
        lower = []
        while True:
            lower.append(pts[i])
            if i == right:
                break
            i = (i + 1) % self.n

        # build upper chain (left → right)
        i = left
        upper = []
        while True:
            upper.append(pts[i])
            if i == right:
                break
            i = (i - 1 + self.n) % self.n

        self.lower = lower
        self.upper = upper

        # extract x-coordinates
        self.xs = sorted(set(x for x, _ in pts))

    # -----------------------------
    # Step 2: linear interpolation
    # -----------------------------
    def _interp(self, p1, p2, x):
        (x1, y1), (x2, y2) = p1, p2
        if x1 == x2:
            return y1
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    # -----------------------------
    # Step 3: evaluate chain at x
    # -----------------------------
    def _build_x_index(self, chain):
        xs = [p[0] for p in chain]
        return xs

    def _y_at(self, chain, xs, x, ptr):
        """
        Move pointer to correct segment and interpolate
        ptr is updated externally for amortized O(1)
        """
        while ptr + 1 < len(chain) and xs[ptr + 1] < x:
            ptr += 1

        # ensure segment contains x
        if ptr + 1 < len(chain):
            return self._interp(chain[ptr], chain[ptr + 1], x), ptr
        else:
            return chain[ptr][1], ptr

    # -----------------------------
    # Step 4: main algorithm
    # -----------------------------
    def max_rectangle_area(self):
        xs = self.xs

        upper = self.upper
        lower = self.lower

        upper_x = [p[0] for p in upper]
        lower_x = [p[0] for p in lower]

        best = 0.0

        for i in range(len(xs)):
            x1 = xs[i]

            top_min = float('inf')
            bot_max = -float('inf')

            # reset pointers for chains
            up_ptr = 0
            low_ptr = 0

            # advance pointers to x1
            while up_ptr + 1 < len(upper) and upper_x[up_ptr + 1] < x1:
                up_ptr += 1
            while low_ptr + 1 < len(lower) and lower_x[low_ptr + 1] < x1:
                low_ptr += 1

            for j in range(i + 1, len(xs)):
                x2 = xs[j]

                # evaluate upper
                while up_ptr + 1 < len(upper) and upper_x[up_ptr + 1] < x2:
                    up_ptr += 1
                y_top = self._interp(upper[up_ptr], upper[min(up_ptr + 1, len(upper)-1)], x2)

                # evaluate lower
                while low_ptr + 1 < len(lower) and lower_x[low_ptr + 1] < x2:
                    low_ptr += 1
                y_bot = self._interp(lower[low_ptr], lower[min(low_ptr + 1, len(lower)-1)], x2)

                # update constraints
                top_min = min(top_min, y_top)
                bot_max = max(bot_max, y_bot)

                height = top_min - bot_max
                if height > 0:
                    area = height * (x2 - x1)
                    if area > best:
                        best = area

        return best